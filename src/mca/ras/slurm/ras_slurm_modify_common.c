/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      Barcelona Supercomputing Center (BSC-CNS).
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "prte_config.h"
#include "constants.h"
#include "types.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "src/runtime/prte_globals.h"
#include "src/runtime/prte_wait.h"

#include "ras_slurm.h"
#include "src/mca/common/slurm/common_slurm.h"
#include "src/mca/ras/base/base.h"

/* The wait status reported for a command whose child could not be waited
 * for: an exit of 127, as a shell reports a command it could not run */
#define PRTE_SLURM_CMD_LOST_STATUS (127 << 8)

/* Bound on how long finalize waits for a killed child to become reapable */
#define PRTE_SLURM_REAP_WAIT_USEC 1000000    /* 1 sec */
#define PRTE_SLURM_REAP_POLL_USEC 10000      /* 10 ms */

/* A Slurm command in flight */
typedef struct {
    pmix_list_item_t super;
    pid_t pid;
    FILE *out;
    prte_ras_slurm_cmd_cbfunc_t cbfunc;
    pmix_object_t *cbdata;
} prte_ras_slurm_cmd_t;

static void cmd_con(prte_ras_slurm_cmd_t *p)
{
    p->pid = -1;
    p->out = NULL;
    p->cbfunc = NULL;
    p->cbdata = NULL;
}

static void cmd_des(prte_ras_slurm_cmd_t *p)
{
    if (NULL != p->out) {
        fclose(p->out);
    }

    if (NULL != p->cbdata) {
        PMIX_RELEASE(p->cbdata);
    }
}

PMIX_CLASS_INSTANCE(prte_ras_slurm_cmd_t, pmix_list_item_t, cmd_con, cmd_des);

/* The job a kill_job callback reports on, and what to run after */
typedef struct {
    pmix_object_t super;
    char *job_id;
    prte_ras_slurm_kill_cbfunc_t cbfunc;
    pmix_object_t *cbdata;
} prte_ras_slurm_kill_t;

static void kill_con(prte_ras_slurm_kill_t *p)
{
    p->job_id = NULL;
    p->cbfunc = NULL;
    p->cbdata = NULL;
}

static void kill_des(prte_ras_slurm_kill_t *p)
{
    free(p->job_id);

    if (NULL != p->cbdata) {
        PMIX_RELEASE(p->cbdata);
    }
}

PMIX_CLASS_INSTANCE(prte_ras_slurm_kill_t, pmix_object_t, kill_con, kill_des);

static pmix_list_t prte_ras_slurm_cmds = PMIX_LIST_STATIC_INIT(prte_ras_slurm_cmds);
/* Set once finalize has begun: nothing reaps a child after that */
static bool cmds_inline = false;

/**
 * @brief Can we serve an elastic allocation request?
 *
 * The whole modify surface -- extend, release and cancel -- rests on reading
 * "scontrol show job <id> --json", and three independent things can stop it.
 * Each of the three entry points asks here first, so a request is refused
 * cleanly rather than half-served, and so the refusal names the actual
 * reason.
 *
 * Two of the three are settled at configure time and folded into one flag by
 * configure.m4: jansson, and whether the Slurm found there was new enough.
 * The third is only knowable here: PRRTE is very often built somewhere other
 * than where it runs -- a build node, a container image, a shared filesystem
 * older than the cluster around it -- so the version that matters is the one
 * on THIS machine, which src/mca/common/slurm probes once and both Slurm
 * components read.
 *
 * The reasons are reported separately on purpose.  They used to share one
 * message naming jansson, which is a plain falsehood on a machine that has
 * jansson and a Slurm too old to emit the schema the parser reads -- and
 * that machine is every stock Ubuntu 24.04, so it is not a corner case.
 *
 * @param[in] quiet  Suppress the diagnostic on refusal (e.g. when the caller
 *                   just wants the answer, not a report to the user).
 *
 * @return true when the request can be served, false (having said why,
 *         unless quiet) when it cannot.
 */
bool prte_ras_slurm_have_extensions(bool quiet)
{
    const prte_common_slurm_version_t *slurm;

#if PRTE_HAVE_SLURM_EXTENSIONS
    if (!prte_ras_slurm_have_jansson()) {
        if (!quiet) {
            pmix_output(0, "ras:slurm:modify: "
                        "Jansson support is required but not enabled in this build");
        }
        return false;
    }

    /* The build is capable; is the scheduler in front of us?  A Slurm we
     * cannot interrogate is NOT treated as too old: this component only runs
     * with SLURM_JOBID set, so something launched us, and refusing on a
     * failed popen would turn a transient into a policy. */
    slurm = prte_common_slurm_version();
    if (slurm->available && !slurm->extended) {
        if (!quiet) {
            pmix_output(0, "ras:slurm:modify: this Slurm (%s) is older than %s, "
                        "whose JSON schema the allocation extensions read. The "
                        "request cannot be served; extend/release/cancel need a "
                        "newer Slurm.",
                        slurm->version, PRTE_SLURM_MIN_EXT_VERSION);
        }
        return false;
    }
    return true;
#else
    /* Say what is true of the build, and let the reader draw the conclusion.
     * Naming a single cause here would be wrong at least a third of the time:
     * the extensions are switched off by an old Slurm, by a missing jansson,
     * or by --disable-slurm-extensions, and configure folds all three into
     * one flag. */
    slurm = prte_common_slurm_version();
    if (!quiet) {
        pmix_output(0, "ras:slurm:modify: the Slurm elastic allocation extensions "
                    "were not built into this PRRTE. They need jansson and Slurm "
                    "%s or later; this tree was configured against Slurm %s, and "
                    "is running under Slurm %s. Rebuild with "
                    "--enable-slurm-extensions (and --with-jansson) to enable "
                    "them.",
                    PRTE_SLURM_MIN_EXT_VERSION, PRTE_SLURM_VERSION_STRING,
                    slurm->version);
    }
    return false;
#endif
}

void prte_ras_slurm_run_cmd_init(void)
{
    PMIX_CONSTRUCT(&prte_ras_slurm_cmds, pmix_list_t);
    cmds_inline = false;
}

void prte_ras_slurm_reap(pid_t pid)
{
    for (uint64_t waited = 0; PRTE_SLURM_REAP_WAIT_USEC > waited;
         waited += PRTE_SLURM_REAP_POLL_USEC) {
        struct timespec tp = {0, PRTE_SLURM_REAP_POLL_USEC * 1000};
        pid_t reaped = waitpid(pid, NULL, WNOHANG);

        if (0 > reaped && EINTR == errno) {
            continue;
        }

        /* Reaped, or never ours to wait for */
        if (0 != reaped) {
            break;
        }

        nanosleep(&tp, NULL);
    }
}

/* Wait for a child the event loop will not reap */
static int cmd_wait_inline(pid_t pid)
{
    int status;

    while (0 > waitpid(pid, &status, 0)) {
        if (EINTR != errno) {
            return PRTE_SLURM_CMD_LOST_STATUS;
        }
    }

    return status;
}

void prte_ras_slurm_run_cmd_finalize(void)
{
    prte_ras_slurm_cmd_t *cmd;

    cmds_inline = true;

    /* Waited for, not killed: a scancel in flight is the only one its job
     * gets. Nobody is called back: the requests behind these are dropped by
     * their own finalize, after pmix_server_finalize() has run */
    while (NULL != (cmd = (prte_ras_slurm_cmd_t *)
                              pmix_list_remove_first(&prte_ras_slurm_cmds))) {
        cmd_wait_inline(cmd->pid);
        /* Its prte_wait_tracker_t is left holding a dangling cbdata, which is
         * what cmd_wait_cb's check is for */
        PMIX_RELEASE(cmd);
    }
}

static void cmd_complete(prte_ras_slurm_cmd_t *cmd, int status)
{
    rewind(cmd->out);
    cmd->cbfunc(status, cmd->out, cmd->cbdata);
    PMIX_RELEASE(cmd);
}

/*
 * Reap a command's child, and hand its output to the caller.
 *
 * @param[in] cbdata The prte_wait_tracker_t, NOT the data handed to
 *                   prte_wait_cb - that is at its cbdata member.
 */
static void cmd_wait_cb(int fd, short args, void *cbdata)
{
    prte_wait_tracker_t *t2 = (prte_wait_tracker_t *) cbdata;
    prte_ras_slurm_cmd_t *cmd = (prte_ras_slurm_cmd_t *) t2->cbdata;
    int status = t2->child->exit_code;
    PRTE_HIDE_UNUSED_PARAMS(fd, args);

    /* The tracker owns the dummy proc; nothing owns cbdata but us */
    PMIX_RELEASE(t2);

    /* Past finalize, cmd has already been killed, reaped and freed */
    if (cmds_inline) {
        return;
    }

    pmix_list_remove_item(&prte_ras_slurm_cmds, &cmd->super);
    cmd_complete(cmd, status);
}

/*
 * Run a shell command with its stdout in an unlinked temporary file, and call
 * cbfunc with its raw wait status and that file, rewound, when it exits. The file is closed when cbfunc returns. cbdata, which may be
 * NULL, is retained until then.
 *
 * Once finalize has begun nothing reaps the child, so the command then runs
 * to completion before this returns.
 *
 * Returns an error, and does not call cbfunc, when the command cannot be
 * started.
 */
int prte_ras_slurm_run_cmd(const char *cmdline, prte_ras_slurm_cmd_cbfunc_t cbfunc,
                           pmix_object_t *cbdata)
{
    prte_ras_slurm_cmd_t *cmd = NULL;
    prte_proc_t *dummy = NULL;
    int err = PRTE_SUCCESS;
    pid_t pid;

    if (NULL == cmdline || NULL == cbfunc) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    PMIX_OUTPUT_VERBOSE((10, prte_ras_base_framework.framework_output,
                         "%s ras:slurm:run_cmd: %s",
                         PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), cmdline));

    cmd = PMIX_NEW(prte_ras_slurm_cmd_t);
    dummy = PMIX_NEW(prte_proc_t);

    if (NULL == cmd || NULL == dummy) {
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* Not the session directory: finalize runs commands after it is gone */
    cmd->out = tmpfile();
    if (NULL == cmd->out) {
        pmix_output(0, "ras:slurm:run_cmd: cannot create a temporary file: %s", strerror(errno));
        err = PRTE_ERR_FILE_OPEN_FAILURE;
        goto cleanup;
    }

    /* Only this command's child may hold it; dup2 clears the flag on its
     * stdout */
    fcntl(fileno(cmd->out), F_SETFD, FD_CLOEXEC);

    pid = fork();

    if (0 > pid) {
        pmix_output(0, "ras:slurm:run_cmd: fork failed: %s", strerror(errno));
        err = PRTE_ERR_SYS_LIMITS_CHILDREN;
        goto cleanup;
    }

    if (0 == pid) {
        int devnull = open("/dev/null", O_RDONLY);

        if (0 <= devnull) {
            dup2(devnull, STDIN_FILENO);
        }

        if (0 > dup2(fileno(cmd->out), STDOUT_FILENO)) {
            _exit(127);
        }

        execl("/bin/sh", "sh", "-c", cmdline, (char *) NULL);
        _exit(127);
    }

    cmd->pid = pid;
    cmd->cbfunc = cbfunc;
    if (NULL != cbdata) {
        PMIX_RETAIN(cbdata);
        cmd->cbdata = cbdata;
    }

    if (cmds_inline) {
        cmd_complete(cmd, cmd_wait_inline(pid));
        cmd = NULL;
        goto cleanup;
    }

    pmix_list_append(&prte_ras_slurm_cmds, &cmd->super);

    dummy->pid = pid;
    /* be sure to mark it as alive so we don't instantly fire */
    PRTE_FLAG_SET(dummy, PRTE_PROC_FLAG_ALIVE);
    prte_wait_cb(dummy, cmd_wait_cb, cmd);
    /* The list owns cmd from here */
    cmd = NULL;

cleanup:
    if (NULL != cmd) {
        PMIX_RELEASE(cmd);
    }

    /* prte_wait_cb retains it */
    if (NULL != dummy) {
        PMIX_RELEASE(dummy);
    }

    return err;
}

static void kill_job_cb(int status, FILE *out, void *cbdata)
{
    prte_ras_slurm_kill_t *kill_data = (prte_ras_slurm_kill_t *) cbdata;
    char err_msg[PRTE_SLURM_ERR_STR_MAX_LEN + 1];

    if (!WIFEXITED(status) || 0 != WEXITSTATUS(status)) {
        if (PRTE_SUCCESS != prte_ras_slurm_drain_cmd_output(out, err_msg, sizeof(err_msg))) {
            err_msg[0] = '\0';
        }

        pmix_output(0, "ras:slurm:kill_job: failed to kill job %s: %s.",
                    kill_data->job_id, err_msg);
    }

    if (NULL != kill_data->cbfunc) {
        kill_data->cbfunc(kill_data->cbdata);
    }
}

/*
 * Cancel a Slurm job using scancel. A failure is reported once scancel exits,
 * and cbfunc, when given, is called after that with cbdata, which is retained
 * until then. Neither happens when an error is returned.
 *
 * @param[in]  slurm_jobid   Null-terminated Slurm job ID string to cancel.
 */
int prte_ras_slurm_kill_job_then(const char *slurm_jobid, prte_ras_slurm_kill_cbfunc_t cbfunc,
                                 pmix_object_t *cbdata)
{
    static const char *cmd_format = "scancel %s 2>&1";
    prte_ras_slurm_kill_t *kill_data;
    char *cmd = NULL;
    int err;

    if (NULL == slurm_jobid) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    /* Make sure the job ID given is something reasonable */
    err = prte_ras_slurm_validate_jobid(slurm_jobid);

    if (PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        return err;
    }

    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                        "%s ras:slurm:kill_job: killing job %s",
                        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), slurm_jobid));

    kill_data = PMIX_NEW(prte_ras_slurm_kill_t);

    if (NULL == kill_data || NULL == (kill_data->job_id = strdup(slurm_jobid)) ||
        0 > asprintf(&cmd, cmd_format, slurm_jobid)) {
        cmd = NULL;
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    kill_data->cbfunc = cbfunc;
    if (NULL != cbdata) {
        PMIX_RETAIN(cbdata);
        kill_data->cbdata = cbdata;
    }

    err = prte_ras_slurm_run_cmd(cmd, kill_job_cb, &kill_data->super);

cleanup:
    if (NULL != kill_data) {
        PMIX_RELEASE(kill_data);
    }

    free(cmd);

    return err;
}

int prte_ras_slurm_kill_job(const char *slurm_jobid)
{
    return prte_ras_slurm_kill_job_then(slurm_jobid, NULL, NULL);
}

/*
 * Check whether a string contains control characters
 *
 * Rejects the string if it contains any control characters
 */
int prte_ras_slurm_token_has_control_chars(const char *s, size_t len, bool *has_control_chars)
{
    if (NULL == s || NULL == has_control_chars) {
        return PRTE_ERR_BAD_PARAM;
    }

    *has_control_chars = false;

    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];

        /* check if control character */
        if (c < 0x20 || c == 0x7f) {
            *has_control_chars = true;
            break;
        }
    }

    return PRTE_SUCCESS;
}

/**
 * @brief Drain command output and optionally capture it.
 *
 * Consumes all data from a command's output stream. When provided, output
 * is populated with command output, potentially truncated to fit the buffer.
 *
 * @param[in] fp Command output stream.
 * @param[out] output Optional output buffer.
 * @param[in] output_size Size of output buffer.
 */
int prte_ras_slurm_drain_cmd_output(FILE *fp, char *output, size_t output_size)
{
    if (NULL == fp) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    if (NULL != output && 0 == output_size) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    char buf[256];
    size_t used = 0;
    bool truncated = false;

    if (NULL != output) {
        output[0] = '\0';
    }

    while (NULL != fgets(buf, sizeof(buf), fp)) {
        if (NULL == output || truncated) {
            continue;
        }

        if (used >= output_size - 1) {
            truncated = true;
            continue;
        }

        size_t len = strlen(buf);
        size_t avail = output_size - used - 1;

        if (len > avail) {
            memcpy(output + used, buf, avail);
            used += avail;
            output[used] = '\0';
            truncated = true;
        } else {
            memcpy(output + used, buf, len);
            used += len;
            output[used] = '\0';
        }
    }

    if (ferror(fp)) {
        PRTE_ERROR_LOG(PRTE_ERR_FILE_READ_FAILURE);
        return PRTE_ERR_FILE_READ_FAILURE;
    }

    if (NULL != output && truncated && output_size >= 4) {
        strcpy(output + output_size - 4, "...");
    }

    return PRTE_SUCCESS;
}
