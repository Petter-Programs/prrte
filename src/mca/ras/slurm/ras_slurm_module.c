/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011-2017 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2013-2020 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2013-2020 Intel, Inc.  All rights reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * Copyright (c) 2021-2026 Nanook Consulting  All rights reserved.
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

#include <ctype.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef HAVE_NETINET_IN_H
#    include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#    include <arpa/inet.h>
#endif
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_JANSSON
#   include <jansson.h>
#endif

#include "src/include/prte_socket_errno.h"
#include "src/util/pmix_argv.h"
#include "src/util/pmix_net.h"
#include "src/util/pmix_output.h"

#include "src/mca/errmgr/errmgr.h"
#include "src/mca/rmaps/base/base.h"
#include "src/mca/state/state.h"
#include "src/runtime/prte_globals.h"
#include "src/util/name_fns.h"
#include "src/util/pmix_show_help.h"

#include "ras_slurm.h"
#include "src/mca/ras/base/base.h"

#define PRTE_SLURM_DYN_MAX_SIZE 256
#define PRTE_SLURM_JOB_INFO_MAX_SIZE (1 * 1024 * 1024)
#define PRTE_SLURM_JOB_ID_MAX_LEN 20
#define PRTE_SLURM_ERR_STR_MAX_SIZE 256
#define PRTE_SLURM_MAX_SBATCH_ARGS 32
#define PRTE_SLURM_MAX_THREADS_PER_CORE 32
#define PRTE_SLURM_MAX_CORE_COUNT 4096

/*
 * API functions
 */
static int init(void);
static int prte_ras_slurm_allocate(prte_job_t *jdata, pmix_list_t *nodes);
static void deallocate(prte_job_t *jdata, prte_app_context_t *app);
static void modify(prte_pmix_server_req_t *req);
static int prte_ras_slurm_finalize(void);

/*
 * RAS slurm module
 */
prte_ras_base_module_t prte_ras_slurm_module = {
    .init = init,
    .allocate = prte_ras_slurm_allocate,
    .deallocate = deallocate,
    .modify = modify,
    .finalize = prte_ras_slurm_finalize
};

/* Local functions */
static int prte_ras_slurm_discover(char *regexp, char *tasks_per_node, pmix_list_t *nodelist);
static int prte_ras_slurm_parse_ranges(char *base, char *ranges, char ***nodelist);
static int prte_ras_slurm_parse_range(char *base, char *range, char ***nodelist);
static int prte_ras_slurm_token_has_control_chars(const char *s, size_t len, bool *has_control_chars);
static int prte_ras_slurm_wipe_dyn_hashtable(pmix_hash_table_t *table);
static int prte_ras_slurm_extract_job_fields(pmix_hash_table_t *values_table);
static int prte_ras_slurm_make_sbatch_arg(pmix_hash_table_t *fields, const char *field_name, const char *field_format, bool obj_num, int *argc, char **argv);
static int prte_ras_slurm_launch_expander_job(pmix_hash_table_t *fields);
static int prte_ras_slurm_exec_sbatch(char * const *argv, char *job_id);
static int prte_ras_slurm_add_modified_resources(const char *slurm_jobid, pmix_list_t *node_list);
static int prte_ras_slurm_wait_resources(const char *slurm_jobid);
static int prte_ras_slurm_kill_job(const char *slurm_jobid, char *err_msg);
static int prte_ras_slurm_validate_jobid(const char *slurm_jobid);
#ifdef HAVE_JANSSON
static int prte_ras_slurm_get_json_numobj_field(json_t *job, const char *key, pmix_hash_table_t *values_table);
static int prte_ras_slurm_extract_job_fields_jansson(pmix_hash_table_t *values_table);
static int prte_ras_slurm_add_modified_resources_jansson(const char *slurm_jobid, pmix_list_t *node_list);
static int prte_ras_slurm_wait_resources_jansson(const char *slurm_jobid);
static int prte_ras_slurm_get_jobinfo_jansson(const char *slurm_jobid, json_t **job_info_out);
#endif

/* Slurm sbatch parameters formats */
static const char *account_format   = "--account=%s";
static const char *partition_format = "--partition=%s";
static const char *qos_format       = "--qos=%s";
static const char *cwd_format       = "--chdir=%s";
static const char *mem_per_cpu_format  = "--mem-per-cpu=%s";
static const char *mem_per_node_format = "--mem=%s";
static const char *time_format = "--time=%s";
static const char *nodes_format = "--nodes=%s";
static const char *threads_per_core_format = "--threads-per-core=%s";

#ifdef HAVE_JANSSON
/* Bounded reader for Slurm JSON output */
typedef struct {
    FILE *fp;
    size_t remaining;
    bool truncated;
    bool io_error;
} jansson_limited_reader_t;
#endif

static char const * const unset_num_marker = "none";  /* Marker for numbers with set: false */
static char const * const infinite_num_marker = "inf"; /* Marker for numbers with set: true and infinite: true */

/* Fields to parse from Slurm JSON */

static char const * const jobs_field = "jobs";

enum slurm_str_field {
    STR_ACCOUNT,
    STR_PARTITION,
    STR_QOS,
    STR_CWD,
    STR_FIELD_COUNT
};

static const char *const str_fields[STR_FIELD_COUNT] = {
    [STR_ACCOUNT]   = "account",
    [STR_PARTITION] = "partition",
    [STR_QOS]       = "qos",
    [STR_CWD]       = "current_working_directory",
};

enum slurm_num_obj_field {
    NUM_OBJ_MEMORY_PER_CPU,
    NUM_OBJ_MEMORY_PER_NODE,
    NUM_OBJ_TIME_LIMIT,
    NUM_OBJ_THREADS_PER_CORE,
    NUM_OBJ_FIELD_COUNT
};

static const char *const num_obj_fields[NUM_OBJ_FIELD_COUNT] = {
    [NUM_OBJ_MEMORY_PER_CPU]  = "memory_per_cpu",
    [NUM_OBJ_MEMORY_PER_NODE] = "memory_per_node",
    [NUM_OBJ_TIME_LIMIT] = "time_limit",
    [NUM_OBJ_THREADS_PER_CORE] = "threads_per_core",
};

enum slurm_num_obj_subfield {
    NUM_OBJ_SUBFIELD_SET,
    NUM_OBJ_SUBFIELD_INFINITE,
    NUM_OBJ_SUBFIELD_NUMBER,
    NUM_OBJ_SUBFIELD_COUNT
};

static const char *const num_obj_subfields[NUM_OBJ_SUBFIELD_COUNT] = {
    [NUM_OBJ_SUBFIELD_SET]      = "set",
    [NUM_OBJ_SUBFIELD_INFINITE] = "infinite",
    [NUM_OBJ_SUBFIELD_NUMBER]   = "number",
};

/* Fields PRRTE records or retrieves from Slurm */

enum record_job_data_field {
    PRTE_JOB_DATA_NODES,
    PRTE_JOB_DATA_JOB_ID,
    PRTE_JOB_DATA_COUNT
};

static const char *const record_job_data_fields[PRTE_JOB_DATA_COUNT] = {
    [PRTE_JOB_DATA_NODES]      = "nodes",
    [PRTE_JOB_DATA_JOB_ID]      = "job_id",
};

static const size_t total_fields_len = STR_FIELD_COUNT + NUM_OBJ_FIELD_COUNT + PRTE_JOB_DATA_COUNT;

static bool check_taint(char *name, char *evar)
{
    int n;

    for (n=0; n < prte_mca_ras_slurm_component.max_length; n++) {
        if ('\0' == evar[n]) {
            return false;
        }
    }

    pmix_show_help("help-ras-slurm.txt", "tainted-envar", true,
                   name, prte_mca_ras_slurm_component.max_length);
    return true;
}

/* init the module */
static int init(void)
{
    return PRTE_SUCCESS;
}

/**
 * Discover available (pre-allocated) nodes.  Allocate the
 * requested number of nodes/process slots to the job.
 *
 */
static int prte_ras_slurm_allocate(prte_job_t *jdata, pmix_list_t *nodes)
{
    int ret, cpus_per_task;
    char *regexp;
    char *tasks_per_node, *node_tasks;
    char *tmp;
    char *slurm_jobid;
    PRTE_HIDE_UNUSED_PARAMS(jdata);

    if (NULL == (slurm_jobid = getenv("SLURM_JOBID"))) {
        return PRTE_ERR_TAKE_NEXT_OPTION;
    }

    regexp = getenv("SLURM_NODELIST");
    if (NULL == regexp) {
        pmix_show_help("help-ras-slurm.txt", "slurm-env-var-not-found", 1, "SLURM_NODELIST");
        return PRTE_ERR_NOT_FOUND;
    }
    // check for length violation - untaint the envar value
    if (check_taint("SLURM_NODELIST", regexp)) {
        return PRTE_ERR_BAD_PARAM;
    }

    if (prte_mca_ras_slurm_component.use_all) {
        /* this is an oddball case required for debug situations where
         * a tool is started that will then call mpirun. In this case,
         * Slurm will assign only 1 tasks/per node to the tool, but
         * we want mpirun to use the entire allocation. They don't give
         * us a specific variable for this purpose, so we have to fudge
         * a bit - but this is a special edge case, and we'll live with it */
        tasks_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
        if (NULL == tasks_per_node) {
            /* couldn't find any version - abort */
            pmix_show_help("help-ras-slurm.txt", "slurm-env-var-not-found", 1,
                           "SLURM_JOB_CPUS_PER_NODE");
            return PRTE_ERR_NOT_FOUND;
        }
        if (check_taint("SLURM_JOB_CPUS_PER_NODE", tasks_per_node)) {
            return PRTE_ERR_BAD_PARAM;
        }

        node_tasks = strdup(tasks_per_node);
        if (NULL == node_tasks) {
            PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
            return PRTE_ERR_OUT_OF_RESOURCE;
        }
        cpus_per_task = 1;

    } else {
        /* get the number of process slots we were assigned on each node */
        tasks_per_node = getenv("SLURM_TASKS_PER_NODE");
        if (NULL == tasks_per_node) {
            /* couldn't find any version - abort */
            pmix_show_help("help-ras-slurm.txt", "slurm-env-var-not-found", 1,
                           "SLURM_TASKS_PER_NODE");
            return PRTE_ERR_NOT_FOUND;
        }
        if (check_taint("SLURM_TASKS_PER_NODE", tasks_per_node)) {
            return PRTE_ERR_BAD_PARAM;
        }

        node_tasks = strdup(tasks_per_node);
        if (NULL == node_tasks) {
            PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
            return PRTE_ERR_OUT_OF_RESOURCE;
        }

        /* get the number of CPUs per task that the user provided to slurm */
        tmp = getenv("SLURM_CPUS_PER_TASK");
        if (NULL != tmp) {
            if (check_taint("SLURM_CPUS_PER_TASK", tmp)) {
                free(node_tasks);
                return PRTE_ERR_BAD_PARAM;
            }
            cpus_per_task = atoi(tmp);
            if (0 >= cpus_per_task) {
                pmix_output(0,
                            "ras:slurm:allocate: Got bad value from SLURM_CPUS_PER_TASK. "
                            "Variable was: %s\n",
                            tmp);
                PRTE_ERROR_LOG(PRTE_ERROR);
                free(node_tasks);
                return PRTE_ERROR;
            }
        } else {
            cpus_per_task = 1;
        }
    }

    ret = prte_ras_slurm_discover(regexp, node_tasks, nodes);
    free(node_tasks);
    if (PRTE_SUCCESS != ret) {
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                             "%s ras:slurm:allocate: discover failed!",
                             PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        return ret;
    }
    /* record the number of allocated nodes */
    prte_num_allocated_nodes = pmix_list_get_size(nodes);

    /* All done */

    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                         "%s ras:slurm:allocate: success", PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
    return PRTE_SUCCESS;
}

static void deallocate(prte_job_t *jdata, prte_app_context_t *app)
{
    PRTE_HIDE_UNUSED_PARAMS(jdata, app);
    return;
}

static void modify(prte_pmix_server_req_t *req) {

    int err = PRTE_SUCCESS;
    int pmix_err = PMIX_SUCCESS;

    pmix_hash_table_t slurm_jobfields;
    pmix_list_t added_nodes;

    bool have_slurm_jobfields = false;
    bool have_added_nodes = false;
    
    char *nodes_string = NULL;

    if(PMIX_ALLOC_EXTEND == req->allocdir) {
     
        PMIX_CONSTRUCT(&slurm_jobfields, pmix_hash_table_t);

        have_slurm_jobfields = true;

        pmix_err = pmix_hash_table_init(&slurm_jobfields, total_fields_len);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_status(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
        
        err = prte_ras_slurm_extract_job_fields(&slurm_jobfields);

        if(PRTE_SUCCESS != err) {
            goto cleanup;
        }

        uint32_t num_nodes;
        bool found = false;

        for (size_t i = 0; i < req->ninfo; ++i) {

            if (0 == strcmp(req->info[i].key, PMIX_NUM_NODES)) {

                if (req->info[i].value.type != PMIX_UINT32) {
                    err = PRTE_ERR_BAD_PARAM;
                    goto cleanup;
                }
            }

            num_nodes = req->info[i].value.data.uint32;
            found = true;
            break;
        }

        if(!found) {
            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                    "%s ras:slurm:modify: modify request invalid",
                    PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));

            err = PRTE_ERR_REQUEST;
            goto cleanup;
        }

        int rc = asprintf(&nodes_string, "%" PRIu32, num_nodes);
        
        if(0 > rc) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        pmix_err = pmix_hash_table_set_value_ptr(&slurm_jobfields, record_job_data_fields[PRTE_JOB_DATA_NODES],
                                strlen(record_job_data_fields[PRTE_JOB_DATA_NODES]), (void*)nodes_string);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_status(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        err = prte_ras_slurm_launch_expander_job(&slurm_jobfields);

        if(PRTE_SUCCESS != err) {
            goto cleanup;
        }

        char *job_id;
        pmix_err = pmix_hash_table_get_value_ptr(&slurm_jobfields, record_job_data_fields[PRTE_JOB_DATA_JOB_ID],
                        strlen(record_job_data_fields[PRTE_JOB_DATA_JOB_ID]), (void**)&job_id);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_status(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        err = prte_ras_slurm_wait_resources(job_id);
        
        if(PRTE_SUCCESS != err) {
            goto cleanup;
        }

        PMIX_CONSTRUCT(&added_nodes, pmix_list_t);

        have_added_nodes = true;

        err = prte_ras_slurm_add_modified_resources(job_id, &added_nodes);

        if(PRTE_SUCCESS != err) {
            goto cleanup;
        }


    }

    cleanup:

    free(nodes_string);

    if(have_slurm_jobfields) {
        prte_ras_slurm_wipe_dyn_hashtable(&slurm_jobfields);
        PMIX_DESTRUCT(&slurm_jobfields);
    }

    if(have_added_nodes) {
        PMIX_DESTRUCT(&added_nodes);
    }

    req->status = err;

    if(err == PRTE_SUCCESS) {
        req->pstatus = PMIX_SUCCESS;
    } else {
        req->pstatus = PMIX_ERROR;
    }

    return;
}

static int prte_ras_slurm_finalize(void)
{
    return PRTE_SUCCESS;
}

/**
 * Discover the available resources.
 *
 * In order to fully support slurm, we need to be able to handle
 * node regexp/task_per_node strings such as:
 * foo,bar    5,3
 * foo        5
 * foo[2-10,12,99-105],bar,foobar[3-11] 2(x10),5,100(x16)
 *
 * @param *regexp A node regular expression from SLURM (i.e. SLURM_NODELIST)
 * @param *tasks_per_node A tasks per node expression from SLURM
 *                        (i.e. SLURM_TASKS_PER_NODE)
 * @param *nodelist A list which has already been constucted to return
 *                  the found nodes in
 */
static int prte_ras_slurm_discover(char *regexp, char *tasks_per_node, pmix_list_t *nodelist)
{
    int i, j, len, ret, count, reps, num_nodes;
    char *base, **names = NULL;
    char *begptr, *endptr, *orig;
    int *slots;
    bool found_range = false;
    bool more_to_come = false;

    orig = base = strdup(regexp);
    if (NULL == base) {
        PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                         "%s ras:slurm:allocate:discover: checking nodelist: %s",
                         PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), regexp));

    do {
        /* Find the base */
        len = strlen(base);
        for (i = 0; i <= len; ++i) {
            if (base[i] == '[') {
                /* we found a range. this gets dealt with below */
                base[i] = '\0';
                found_range = true;
                break;
            }
            if (base[i] == ',') {
                /* we found a singleton node, and there are more to come */
                base[i] = '\0';
                found_range = false;
                more_to_come = true;
                break;
            }
            if (base[i] == '\0') {
                /* we found a singleton node */
                found_range = false;
                more_to_come = false;
                break;
            }
        }
        if (i == 0) {
            /* we found a special character at the beginning of the string */
            pmix_show_help("help-ras-slurm.txt", "slurm-env-var-bad-value", 1, regexp,
                           tasks_per_node, "SLURM_NODELIST");
            PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
            free(orig);
            return PRTE_ERR_BAD_PARAM;
        }

        if (found_range) {
            /* If we found a range, now find the end of the range */
            for (j = i; j < len; ++j) {
                if (base[j] == ']') {
                    base[j] = '\0';
                    break;
                }
            }
            if (j >= len) {
                /* we didn't find the end of the range */
                pmix_show_help("help-ras-slurm.txt", "slurm-env-var-bad-value", 1, regexp,
                               tasks_per_node, "SLURM_NODELIST");
                PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
                free(orig);
                return PRTE_ERR_BAD_PARAM;
            }

            ret = prte_ras_slurm_parse_ranges(base, base + i + 1, &names);
            if (PRTE_SUCCESS != ret) {
                pmix_show_help("help-ras-slurm.txt", "slurm-env-var-bad-value", 1, regexp,
                               tasks_per_node, "SLURM_NODELIST");
                PRTE_ERROR_LOG(ret);
                free(orig);
                return ret;
            }
            if (base[j + 1] == ',') {
                more_to_come = true;
                base = &base[j + 2];
            } else {
                more_to_come = false;
            }
        } else {
            /* If we didn't find a range, just add the node */

            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                                 "%s ras:slurm:allocate:discover: found node %s",
                                 PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), base));

            if (PRTE_SUCCESS != (ret = PMIx_Argv_append_nosize(&names, base))) {
                PRTE_ERROR_LOG(ret);
                free(orig);
                return ret;
            }
            /* set base equal to the (possible) next base to look at */
            base = &base[i + 1];
        }
    } while (more_to_come);

    free(orig);

    num_nodes = PMIx_Argv_count(names);

    /* Find the number of slots per node */

    slots = malloc(sizeof(int) * num_nodes);
    if (NULL == slots) {
        PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
        return PRTE_ERR_OUT_OF_RESOURCE;
    }
    memset(slots, 0, sizeof(int) * num_nodes);

    orig = begptr = strdup(tasks_per_node);
    if (NULL == begptr) {
        PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
        free(slots);
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    j = 0;
    while (begptr) {
        count = strtol(begptr, &endptr, 10);
        if ((endptr[0] == '(') && (endptr[1] == 'x')) {
            reps = strtol((endptr + 2), &endptr, 10);
            if (endptr[0] == ')') {
                endptr++;
            }
        } else {
            reps = 1;
        }

        /**
         * TBP: it seems like it would be an error to have more slot
         * descriptions than nodes. Turns out that this valid, and SLURM will
         * return such a thing. For instance, if I did:
         * srun -A -N 30 -w odin001
         * I would get SLURM_NODELIST=odin001 SLURM_TASKS_PER_NODE=4(x30)
         * That is, I am allocated 30 nodes, but since I only requested
         * one specific node, that's what is in the nodelist.
         * I'm not sure this is what users would expect, but I think it is
         * more of a SLURM issue than a prte issue, since SLURM is OK with it,
         * I'm ok with it
         */
        for (i = 0; i < reps && j < num_nodes; i++) {
            slots[j++] = count;
        }

        if (*endptr == ',') {
            begptr = endptr + 1;
        } else if (*endptr == '\0' || j >= num_nodes) {
            break;
        } else {
            pmix_show_help("help-ras-slurm.txt", "slurm-env-var-bad-value", 1, regexp,
                           tasks_per_node, "SLURM_TASKS_PER_NODE");
            PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
            free(slots);
            free(orig);
            return PRTE_ERR_BAD_PARAM;
        }
    }

    free(orig);

    /* Convert the argv of node names to a list of node_t's */

    for (i = 0; NULL != names && NULL != names[i]; ++i) {
        prte_node_t *node;

        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                             "%s ras:slurm:allocate:discover: adding node %s (%d slot%s)",
                             PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), names[i], slots[i],
                             (1 == slots[i]) ? "" : "s"));

        node = PMIX_NEW(prte_node_t);
        if (NULL == node) {
            PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
            free(slots);
            return PRTE_ERR_OUT_OF_RESOURCE;
        }
        node->name = strdup(names[i]);
        node->state = PRTE_NODE_STATE_UP;
        node->slots_inuse = 0;
        node->slots_max = 0;
        node->slots = slots[i];
        pmix_list_append(nodelist, &node->super);
    }
    free(slots);
    PMIx_Argv_free(names);

    /* All done */
    return ret;
}

/*
 * Parse one or more ranges in a set
 *
 * @param base     The base text of the node name
 * @param *ranges  A pointer to a range. This can contain multiple ranges
 *                 (i.e. "1-3,10" or "5" or "9,0100-0130,250")
 * @param ***names An argv array to add the newly discovered nodes to
 */
static int prte_ras_slurm_parse_ranges(char *base, char *ranges, char ***names)
{
    int i, len, ret;
    char *start, *orig;

    /* Look for commas, the separator between ranges */

    len = strlen(ranges);
    for (orig = start = ranges, i = 0; i < len; ++i) {
        if (',' == ranges[i]) {
            ranges[i] = '\0';
            ret = prte_ras_slurm_parse_range(base, start, names);
            if (PRTE_SUCCESS != ret) {
                PRTE_ERROR_LOG(ret);
                return ret;
            }
            start = ranges + i + 1;
        }
    }

    /* Pick up the last range, if it exists */

    if (start < orig + len) {

        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                             "%s ras:slurm:allocate:discover: parse range %s (2)",
                             PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), start));

        ret = prte_ras_slurm_parse_range(base, start, names);
        if (PRTE_SUCCESS != ret) {
            PRTE_ERROR_LOG(ret);
            return ret;
        }
    }

    /* All done */
    return PRTE_SUCCESS;
}

/*
 * Parse a single range in a set and add the full names of the nodes
 * found to the names argv
 *
 * @param base     The base text of the node name
 * @param *ranges  A pointer to a single range. (i.e. "1-3" or "5")
 * @param ***names An argv array to add the newly discovered nodes to
 */
static int prte_ras_slurm_parse_range(char *base, char *range, char ***names)
{
    char *str, temp1[BUFSIZ];
    size_t i, j, start, end;
    size_t base_len, len, num_len;
    size_t num_str_len;
    bool found;
    int ret;

    len = strlen(range);
    base_len = strlen(base);
    /* Silence compiler warnings; start and end are always assigned
       properly, below */
    start = end = 0;

    /* Look for the beginning of the first number */

    for (found = false, i = 0; i < len; ++i) {
        if (isdigit((int) range[i])) {
            if (!found) {
                start = atoi(range + i);
                found = true;
                break;
            }
        }
    }
    if (!found) {
        PRTE_ERROR_LOG(PRTE_ERR_NOT_FOUND);
        return PRTE_ERR_NOT_FOUND;
    }

    /* Look for the end of the first number */

    for (found = false, num_str_len = 0; i < len; ++i, ++num_str_len) {
        if (!isdigit((int) range[i])) {
            break;
        }
    }

    /* Was there no range, just a single number? */

    if (i >= len) {
        end = start;
        found = true;
    }

    /* Nope, there was a range.  Look for the beginning of the second
       number */

    else {
        for (; i < len; ++i) {
            if (isdigit((int) range[i])) {
                end = atoi(range + i);
                found = true;
                break;
            }
        }
    }
    if (!found) {
        PRTE_ERROR_LOG(PRTE_ERR_NOT_FOUND);
        return PRTE_ERR_NOT_FOUND;
    }

    /* Make strings for all values in the range */

    len = base_len + num_str_len + 32;
    str = malloc(len);
    if (NULL == str) {
        PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
        return PRTE_ERR_OUT_OF_RESOURCE;
    }
    strcpy(str, base);
    for (i = start; i <= end; ++i) {
        str[base_len] = '\0';
        snprintf(temp1, BUFSIZ - 1, "%lu", (long) i);

        /* Do we need zero pading? */

        if ((num_len = strlen(temp1)) < num_str_len) {
            for (j = base_len; j < base_len + (num_str_len - num_len); ++j) {
                str[j] = '0';
            }
            str[j] = '\0';
        }
        strcat(str, temp1);
        ret = PMIx_Argv_append_nosize(names, str);
        if (PRTE_SUCCESS != ret) {
            PRTE_ERROR_LOG(ret);
            free(str);
            return ret;
        }
    }
    free(str);

    /* All done */
    return PRTE_SUCCESS;
}

/*
 * Check whether a string contains control characters
 *
 * Rejects the string if it contains any control characters
 */
static int prte_ras_slurm_token_has_control_chars(const char *s, size_t len, bool *has_control_chars) {
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

/*
 * Wipe and free() all values in a PMIx hash table.
 *
 * Iterates over all elements in the given PMIx hash table,
 * frees the memory pointed to by each element's value field, and then
 * removes all entries from the table.
 * 
 * This function does not destroy the hash table object itself.
 * 
 * @param table Pointer to the PMIx hash table to be wiped.
 */
static int prte_ras_slurm_wipe_dyn_hashtable(pmix_hash_table_t *table) {

    void *key;
    size_t key_size;
    void *val;
    void *prev_node;
    void *node;

    int pmix_err = PMIX_SUCCESS;

    pmix_err = pmix_hash_table_get_first_key_ptr(table, &key, &key_size, &val, &node);

    if(PMIX_SUCCESS != pmix_err) {
        return prte_pmix_convert_status(pmix_err);
    }
    
    free(val);
    prev_node = node;

    while (PMIX_SUCCESS ==
        pmix_hash_table_get_next_key_ptr(table, &key, &key_size, &val, &prev_node, &node)) {
        free(val);
        prev_node = node;
    }

    pmix_err = pmix_hash_table_remove_all(table);
    return prte_pmix_convert_status(pmix_err);
}

/*
 * Extract SLURM job fields into a PMIx hash table.
 *
 * Wrapper around the Jansson-based implementation when available.
 *
 * @param[in,out] values_table Pointer to a PMIx hash table to populate.
 */
static int prte_ras_slurm_extract_job_fields(pmix_hash_table_t *values_table) {
#ifdef HAVE_JANSSON
    return prte_ras_slurm_extract_job_fields_jansson(values_table);
#else
        pmix_output(0,
                    "ras:slurm:extract_job_fields: feature requires the Jansson "
                    "library to be available in compilation.\n");
    return PRTE_ERR_NOT_SUPPORTED;
#endif
}

/*
 * Add nodes and slots from a given Slurm job to PRRTE.
 *
 * Wrapper around the Jansson-based implementation when available.
 *
 * @param[in] slurm_jobid Slurm job ID.
 * @param[in,out] node_list A pmix_list_t to add nodes to.
 */
static int prte_ras_slurm_add_modified_resources(const char *slurm_jobid, pmix_list_t *node_list) {
#ifdef HAVE_JANSSON
    return prte_ras_slurm_add_modified_resources_jansson(slurm_jobid, node_list);
#else
    pmix_output(0,
                    "ras:slurm:add_modified_resources: feature requires the Jansson "
                    "library to be available in compilation.\n");
    return PRTE_ERR_NOT_SUPPORTED;
#endif
}

/*
 * @brief Wait until the given Slurm job reaches the RUNNING state.
 *
 * Wrapper around the Jansson-based implementation when available.
 *
 * @param[in] slurm_jobid Slurm job identifier.
 */
static int prte_ras_slurm_wait_resources(const char *slurm_jobid) {
#ifdef HAVE_JANSSON
    return prte_ras_slurm_wait_resources_jansson(slurm_jobid);
#else
    pmix_output(0,
                    "ras:slurm:wait_resources: feature requires the Jansson "
                    "library to be available in compilation.\n");
    return PRTE_ERR_NOT_SUPPORTED;
#endif
}

#ifdef HAVE_JANSSON

/*
 * Extract selected Slurm job fields using JSON and populate a PMIx hash table.
 *
 * Retrieves the SLURM job ID from the environment, queries job information,
 * parses the returned JSON using Jansson, and inserts selected numeric and
 * string fields into the provided hash table.
 *
 * String fields are validated to ensure they do not contain control characters.
 *
 * @param[in,out] values_table Pointer to a PMIx hash table to populate with extracted values.

 * Note: On failure, values_table may be partially populated.
 */
static int prte_ras_slurm_extract_job_fields_jansson(pmix_hash_table_t *values_table) {
    
    if(NULL == values_table) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int err = PRTE_SUCCESS;
    int pmix_err = PMIX_SUCCESS;

    json_t *job = NULL;

    char *slurm_jobid;
    if (NULL == (slurm_jobid = getenv("SLURM_JOBID"))) {
        PRTE_ERROR_LOG(PRTE_ERR_NOT_FOUND);
        return PRTE_ERR_NOT_FOUND;
    }

    /* Read JSON from stream and extract the first and only job 
       in the "jobs" array, taking ownership of the returned json. */
    err = prte_ras_slurm_get_jobinfo_jansson(slurm_jobid, &job);

    if(PRTE_SUCCESS != err) {
        goto cleanup;
    }

    /* We've extracted a valid "jobs" section. now extract the complex numeric
    * fields that have "set", "infinite", and "number" subfields */
    for(size_t i = 0; i < NUM_OBJ_SUBFIELD_COUNT; i++) {
        err = prte_ras_slurm_get_json_numobj_field(job, num_obj_fields[i], values_table);
        if (PRTE_SUCCESS != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    /* Find the string fields and add them to our values table */

    for(size_t i = 0; i < STR_FIELD_COUNT; i++) {

        json_t *str_field = json_object_get(job, str_fields[i]);

        if(NULL == str_field || !json_is_string(str_field)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup; 
        }

        const char *str = json_string_value(str_field);
        size_t str_len = json_string_length(str_field);
        bool has_control_chars; 

        /* Do not accept string if contains control characters */
        err = prte_ras_slurm_token_has_control_chars(str, str_len, &has_control_chars);

        if(PRTE_SUCCESS == err && has_control_chars) {
            err = PRTE_ERR_BAD_PARAM;
        }

        if(PRTE_SUCCESS != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup; 
        }

        char *str_dup = strdup(str);

        if(NULL == str_dup) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            PRTE_ERROR_LOG(err);
            goto cleanup; 
        }

        pmix_err = pmix_hash_table_set_value_ptr(values_table, str_fields[i],
                    strlen(str_fields[i]), str_dup);

        if(PMIX_SUCCESS != pmix_err) {
            free(str_dup);
            err = prte_pmix_convert_status(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

    }

    cleanup:

    if(NULL != job) {
        json_decref(job);
    }

    return err;
}

/*
 * Parse a numeric-object field from JSON and store it as a string in a hash table.
 *
 * Expects the the JSON object at key in job to have
 * "set", "infinite", and "number" fields. Stores the result
 * in values_table:
 * - unset → string determined by unset_num_marker
 * - infinite → string determined by infinite_num_marker
 * - otherwise → numeric value as string
 *
 * @param[in]  job           JSON job object.
 * @param[in]  key           Field name to extract.
 * @param[out] values_table  Destination hash table.
 */
static int prte_ras_slurm_get_json_numobj_field(json_t *job, const char *key, pmix_hash_table_t *values_table) {

    if (NULL == job || NULL == key || NULL == values_table) {
        return PRTE_ERR_BAD_PARAM;
    }

    int pmix_err = PMIX_SUCCESS;

    json_t *field = json_object_get(job, key);
    if (NULL == field || !json_is_object(field)) {
        return PRTE_ERR_JSON_PARSE_FAILURE;
    }

    json_t *set_flag = json_object_get(field, num_obj_subfields[NUM_OBJ_SUBFIELD_SET]);
    if (NULL == set_flag || !json_is_boolean(set_flag)) {
        return PRTE_ERR_JSON_PARSE_FAILURE;
    }

    if (!json_is_true(set_flag)) {
        char *unset_dyn = strdup(unset_num_marker);
        if (NULL == unset_dyn) {
            return PRTE_ERR_OUT_OF_RESOURCE;
        }

        pmix_err = pmix_hash_table_set_value_ptr(values_table, key, strlen(key), unset_dyn);
        if (PMIX_SUCCESS != pmix_err) {
            free(unset_dyn);
            return prte_pmix_convert_status(pmix_err);
        }
        return PRTE_SUCCESS;
    }

    json_t *inf_flag = json_object_get(field, num_obj_subfields[NUM_OBJ_SUBFIELD_INFINITE]);
    if (NULL == inf_flag || !json_is_boolean(inf_flag)) {
        return PRTE_ERR_JSON_PARSE_FAILURE;
    }

    if (json_is_true(inf_flag)) {
        char *inf_dyn = strdup(infinite_num_marker);
        if (NULL == inf_dyn) return PRTE_ERR_OUT_OF_RESOURCE;

        pmix_err = pmix_hash_table_set_value_ptr(values_table, key, strlen(key), inf_dyn);
        if (PMIX_SUCCESS != pmix_err) {
            free(inf_dyn);
            return prte_pmix_convert_status(pmix_err);
        }
        return PRTE_SUCCESS;
    }

    json_t *num_field = json_object_get(field, num_obj_subfields[NUM_OBJ_SUBFIELD_NUMBER]);
    if (NULL == num_field || !json_is_integer(num_field)) {
        return PRTE_ERR_JSON_PARSE_FAILURE;
    }

    json_int_t num = json_integer_value(num_field);
    if (num < 0) {
        return PRTE_ERR_JSON_PARSE_FAILURE;
    }

    char *num_dyn = NULL;
    if (-1 == asprintf(&num_dyn, "%" JSON_INTEGER_FORMAT, num)) {
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    pmix_err = pmix_hash_table_set_value_ptr(values_table, key, strlen(key), num_dyn);
    if (PMIX_SUCCESS != pmix_err) {
        free(num_dyn);
        return prte_pmix_convert_status(pmix_err);
    }

    return PRTE_SUCCESS;
}

/*
 * Fetch and parse Slurm job resource JSON and add allocated nodes and slots.
 *
 * Given a Slurm job ID, this function retrieves the job resource description,
 * validates the expected JSON structure, and creates one node entry for
 * each allocated node in the job.
 *
 * Slot calculation is based on the number of allocated cores found in the
 * socket/core status data, multiplied by the effective threads-per-core value,
 * and capped by cpus.count:
 *
 * slots = min(allocated_cores * threads_per_core, cpus.count)
 *
 * The resulting nodes are inserted into the provided node list.
 *
 * @param[in] slurm_jobid Slurm job ID.
 * @param[in,out] node_list. A pmix_list_t to add nodes to.
 */
static int prte_ras_slurm_add_modified_resources_jansson(const char *slurm_jobid, pmix_list_t *node_list) {

    if(NULL == slurm_jobid || NULL == node_list) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int err = PRTE_SUCCESS;

    err = prte_ras_slurm_validate_jobid(slurm_jobid);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        return err;
    }

    int threads_per_core = 1;

    json_t *root = NULL;

    err = prte_ras_slurm_get_jobinfo_jansson(slurm_jobid, &root);

    if(PRTE_SUCCESS != err) {
        goto cleanup;
    }

    json_t *tpc_obj = json_object_get(root, "threads_per_core");

    if(NULL == tpc_obj || !json_is_object(tpc_obj)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_t *tpc_set_flag = json_object_get(tpc_obj, "set");

    if(NULL == tpc_set_flag || !json_is_boolean(tpc_set_flag)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_t *tpc_inf_flag = json_object_get(tpc_obj, "infinite");

    if(NULL == tpc_inf_flag || !json_is_boolean(tpc_inf_flag)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_t *tpc_val = json_object_get(tpc_obj, "number");

    if(NULL == tpc_val || !json_is_integer(tpc_val)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_int_t tpc = json_integer_value(tpc_val);

    if(json_is_true(tpc_set_flag) && json_is_false(tpc_inf_flag)) {
        if (tpc > 0 && tpc <= PRTE_SLURM_MAX_THREADS_PER_CORE) {
            threads_per_core = (int)tpc;
        } else if (tpc == 0) {
            /* Slurm could set to 0 in some cases */
            threads_per_core = 1;
        } else {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    json_t *job_resources = json_object_get(root, "job_resources");

    if(NULL == job_resources || !json_is_object(job_resources)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_t *nodes = json_object_get(job_resources, "nodes");

    if(NULL == nodes || !json_is_object(nodes)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_t *allocation = json_object_get(nodes, "allocation");

    if (NULL == allocation || !json_is_array(allocation)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    size_t node_idx;
    json_t *node_obj;

    /* Retrieve node names and allocated slots */
    json_array_foreach(allocation, node_idx, node_obj) {
        if (!json_is_object(node_obj)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        json_t *nodename = json_object_get(node_obj, "name");

        if (NULL == nodename || !json_is_string(nodename)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        const char *nodename_string = json_string_value(nodename);

        if (NULL == nodename_string || '\0' == nodename_string[0]) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        json_t *cpu_info = json_object_get(node_obj, "cpus");

        if (NULL == cpu_info || !json_is_object(cpu_info)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        json_t *cpu_count_obj = json_object_get(cpu_info, "count");

        if (NULL == cpu_count_obj || !json_is_integer(cpu_count_obj)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        json_int_t cpu_count_num = json_integer_value(cpu_count_obj);

        if (0 >= cpu_count_num || INT_MAX < cpu_count_num) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        int cpu_max_count = (int)cpu_count_num;

        int core_count = 0;

        json_t *sockets = json_object_get(node_obj, "sockets");

        if(NULL == sockets || !json_is_array(sockets)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup; 
        }

        size_t socket_idx;
        json_t *socket_obj;

        json_array_foreach(sockets, socket_idx, socket_obj) {

            if (!json_is_object(socket_obj)) {
                err = PRTE_ERR_JSON_PARSE_FAILURE;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            json_t *cores = json_object_get(socket_obj, "cores");

            if (NULL == cores || !json_is_array(cores)) {
                err = PRTE_ERR_JSON_PARSE_FAILURE;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            size_t core_idx;
            json_t *core_obj;

            json_array_foreach(cores, core_idx, core_obj) {

                if(!json_is_object(core_obj)) {
                    err = PRTE_ERR_JSON_PARSE_FAILURE;
                    PRTE_ERROR_LOG(err);
                    goto cleanup;
                }

                json_t *statuses = json_object_get(core_obj, "status");
                if (NULL == statuses || !json_is_array(statuses)) {
                    err = PRTE_ERR_JSON_PARSE_FAILURE;
                    PRTE_ERROR_LOG(err);
                    goto cleanup;
                }

                size_t status_idx;
                json_t *status_obj;

                json_array_foreach(statuses, status_idx, status_obj) {

                    if(!json_is_string(status_obj)) {
                        err = PRTE_ERR_JSON_PARSE_FAILURE;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    if (0 == strcmp(json_string_value(status_obj), "ALLOCATED")) {
                        core_count++;

                        if(PRTE_SLURM_MAX_CORE_COUNT < core_count) {
                            err = PRTE_ERR_JSON_PARSE_FAILURE;
                            PRTE_ERROR_LOG(err);
                            goto cleanup;
                        }
                        
                        break;
                    }
                }
            }
        }

        if(0 >= core_count) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        if (core_count > INT_MAX / threads_per_core) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        char *nodename_dyn = strdup(nodename_string);

        if (NULL == nodename_dyn) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            PRTE_ERROR_LOG(err);
            goto cleanup;  
        }

        /* 
        The "cpus" field represents the number of hardware
        threads available. To respect the preferences of the
        original job, we calculate (cores * threads_per_core),
        ensuring it does not exceed the available count as provided
        by the "cpus" field. Note that if threads_per_core is
        unset, infinite, or out of expected bounds, we default to 1.
        If threads_per_core is missing entirely, we error out.
        */

        int slots = core_count * threads_per_core;
        
        if(slots > cpu_max_count) {
            slots = cpu_max_count;
        }

        prte_node_t *node = PMIX_NEW(prte_node_t);
        
        node->state = PRTE_NODE_STATE_UP;
        node->name = nodename_dyn;
        node->slots_inuse = 0;
        node->slots_max = 0;
        node->slots = slots;

        pmix_list_append(node_list, &node->super);

        PMIX_OUTPUT_VERBOSE((20, prte_ras_base_framework.framework_output,
        "%s ras:slurm:add_modified_resources: discovered node %s with "
        "%d slots",
        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), node->name, node->slots)); 
    }

    cleanup:

    if(NULL != root) {
        json_decref(root);
    }

    return err;
}

/*
 * Jansson input callback with read size limiting.
 *
 * Reads data from a FILE stream into the provided buffer, enforcing a
 * maximum total number of bytes that can be consumed. If the limit is
 * reached, the reader is marked as truncated and no further data is read.
 * Intended for use with json_load_callback().
 *
 * @param[out] buffer
 *     Destination buffer for read data.
 * @param[in] buflen
 *     Maximum number of bytes to read into the buffer.
 * @param[in,out] data
 *     Pointer to a jansson_limited_reader_t structure containing the FILE
 *     stream, remaining byte budget, and truncation and error flags.
 *
 * @return Number of bytes read into buffer. Returns 0 when no more data
 *         should be read (EOF or limit reached).
 */
static size_t prte_ras_slurm_jansson_cbfunc(void *buffer, size_t buflen, void *data) {
    jansson_limited_reader_t *reader = data;

    if (reader->remaining == 0) {
        reader->truncated = 1;
        return 0;
    }

    if (buflen > reader->remaining) {
        buflen = reader->remaining;
    }

    size_t len = fread(buffer, 1, buflen, reader->fp);

    if (0 == len && ferror(reader->fp)) {
        reader->io_error = true;
    }

    reader->remaining -= len;
    return len;
}

/*
 * Query Slurm job information and return the job object as Jansson JSON.
 *
 * Executes `scontrol show job <jobid> --json`, parses the resulting JSON,
 * and returns the single job object contained in the response. On success,
 * the returned JSON object is referenced for the caller, who becomes responsible
 * for releasing it with json_decref().
 *
 * @param[in] slurm_jobid
 *     SLURM job ID to query.
 * @param[out] job_info_out
 *     Output pointer receiving the parsed JSON object for the job. Set to
 *     NULL on entry and on failure.
 */
static int prte_ras_slurm_get_jobinfo_jansson(const char *slurm_jobid, json_t **job_info_out) {

    if(NULL == slurm_jobid || NULL == job_info_out) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    *job_info_out = NULL;

    int err = PRTE_SUCCESS;

    /* Make sure the job ID given is within constraints */
    err = prte_ras_slurm_validate_jobid(slurm_jobid);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        return err;
    }

    static const char *cmd_format = "scontrol show job %s --json";
    
    json_error_t json_err;

    json_t *parent_json = NULL;

    FILE *fp = NULL;

    char *cmd = NULL;

    if(0 > asprintf(&cmd, cmd_format, slurm_jobid)) {
        cmd = NULL;
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        return err;
    }

    fp = popen(cmd, "r");

    if(NULL == fp) {
        err = PRTE_ERR_FILE_OPEN_FAILURE;
        goto cleanup;
    }

    jansson_limited_reader_t lr = {
        .fp = fp,
        .remaining = PRTE_SLURM_JOB_INFO_MAX_SIZE,
        .truncated = false,
        .io_error = false
    };

    parent_json = json_load_callback(
        prte_ras_slurm_jansson_cbfunc,
        &lr,
        JSON_REJECT_DUPLICATES,
        &json_err
    );

    int status = pclose(fp);
    fp = NULL;

    if (-1 == status) {
        err = PRTE_ERR_FILE_OPEN_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    if (!WIFEXITED(status) || 0 != WEXITSTATUS(status)) {
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:get_jobinfo_jansson: non-zero exit code (%d) from scontrol command.",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME),
            WIFEXITED(status) ? WEXITSTATUS(status) : -1));
        err = PRTE_ERR_SLURM_QUERY_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    if(!parent_json) {

        if(lr.io_error) {
            err = PRTE_ERR_FILE_READ_FAILURE;
            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:get_jobinfo_jansson: error reading from stream.",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        } else if(lr.truncated) {
            err = PRTE_ERR_MEM_LIMIT_EXCEEDED;
            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:get_jobinfo_jansson: job info JSON was truncated.",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        } else {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:get_jobinfo_jansson: job info JSON parse failed.",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        }

        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* Jobs array: we expect and require exactly one job in the result  */
    json_t *jobs_arr = json_object_get(parent_json, jobs_field);
    if (NULL == jobs_arr || !json_is_array(jobs_arr) || 1 != json_array_size(jobs_arr)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    json_t *job = json_array_get(jobs_arr, 0);
    if (NULL == job || !json_is_object(job)) {
        err = PRTE_ERR_JSON_PARSE_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* Ensure job information is not destroyed */
    json_incref(job);

    *job_info_out = job;

    cleanup:

    if(NULL != fp) {
        pclose(fp);
    }

    if(NULL != parent_json) {
        json_decref(parent_json);
    }

    free(cmd);

    return err;
}

/*
 * Wait until a Slurm job contains the RUNNING state.
 *
 * Polls Slurm job information once per second and inspects
 * the "job_state" JSON field until the job transitions 
 * out of PENDING. The function returns success only if the job
 * reaches RUNNING.
 *
 * @param[in] slurm_jobid
 *     SLURM job ID to monitor.
 */
static int prte_ras_slurm_wait_resources_jansson(const char *slurm_jobid) {
    
    int err = PRTE_SUCCESS;

    err = prte_ras_slurm_validate_jobid(slurm_jobid);

    if(PRTE_SUCCESS != err) {
        return PRTE_ERR_BAD_PARAM;
    }

    json_t *job_info = NULL;

    bool running;
    bool pending;

    do {
        running = false;
        pending = false;
        err = prte_ras_slurm_get_jobinfo_jansson(slurm_jobid, &job_info);

        if(PRTE_SUCCESS != err) {
            goto cleanup;
        }

        json_t *job_states = json_object_get(job_info, "job_state");

        /* A job can have multiple states in Slurm */
        if (NULL == job_states || !json_is_array(job_states)) {
            err = PRTE_ERR_JSON_PARSE_FAILURE;
            PRTE_ERROR_LOG(PRTE_ERR_JSON_PARSE_FAILURE);
            goto cleanup;
        }
        
        size_t i;
        json_t *state_val;

        json_array_foreach(job_states, i, state_val) {

            if(!json_is_string(state_val)) {
                err = PRTE_ERR_JSON_PARSE_FAILURE;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            const char *state = json_string_value(state_val);

            if (strcmp(state, "RUNNING") == 0) {
                running = true;
            }

            else if (strcmp(state, "PENDING") == 0) {
                pending = true;
            }
        }

        json_decref(job_info);
        job_info = NULL;

        /* Should be mutually exclusive */
        if (running && pending) {
            err = PRTE_ERR_SLURM_BAD_JOB_STATUS;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        /* Avoid overloading Slurm with requests */
        if(pending) {
            sleep(1);
        }

    } while (pending);

    if(!running) {
        err = PRTE_ERR_SLURM_BAD_JOB_STATUS;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    cleanup:

    if(NULL != job_info) {
        json_decref(job_info);
    }

    return err;
}

#endif

/*
 * Append a formatted sbatch argument from a pmix hash table field.
 *
 * Looks up a value in the provided hash table and, if present and usable,
 * formats it according to the given format string and appends it to the
 * sbatch argv array.
 *
 * @param[in] fields
 *     Hash table containing job configuration data.
 * @param[in] field_name
 *     Key used to retrieve the value from the hash table.
 * @param[in] field_format
 *     printf-style format string used to construct the sbatch argument.
 * @param[in] obj_num
 *     Indicates whether the field represents a numeric object; enables
 *     filtering of special sentinel values (e.g., "unset", "infinite").
 * @param[in,out] argc
 *     Current argument count. Incremented if an argument is appended.
 * @param[in,out] argv
 *     Argument vector to append to (size PRTE_SLURM_MAX_SBATCH_ARGS+1).
 */
static int prte_ras_slurm_make_sbatch_arg(pmix_hash_table_t *fields,
                                          const char *field_name,
                                          const char *field_format,
                                          bool obj_num,
                                          int *argc,
                                          char **argv
                                          )
{    
    if(NULL == fields || NULL == field_name || NULL == field_format 
    || NULL == argv || NULL == argc || *argc < 0) {
        return PRTE_ERR_BAD_PARAM;
    }

    if(*argc >= PRTE_SLURM_MAX_SBATCH_ARGS) {
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    char *stored_val = NULL;

    int pmix_err = pmix_hash_table_get_value_ptr(fields, field_name,
                        strlen(field_name), (void**)&stored_val);

    if(PMIX_SUCCESS != pmix_err) {
        /* converts PMIX_ERR_NOT_FOUND->PRTE_ERR_NOT_FOUND if not found */
        return prte_pmix_convert_status(pmix_err);
    }

    if(NULL == stored_val || '\0' == stored_val[0]) {
        return PRTE_ERR_DATA_VALUE_NOT_FOUND;
    }

    if(obj_num) {
        /* handle both as just unset for now */
        if(0 == strcmp(stored_val, unset_num_marker)
        || 0 == strcmp(stored_val, infinite_num_marker)) {
            return PRTE_ERR_NOT_FOUND;
        }
    }

    int rc = asprintf(&argv[*argc], field_format, stored_val);

    if(0 > rc) {
        argv[*argc] = NULL;
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    (*argc)++;
    argv[*argc] = NULL;

    return PRTE_SUCCESS;
}

/*
 * Construct and launch a Slurm "expander" job via sbatch.
 *
 * Constructs an sbatch command using parameters stored in the provided
 * hash table. Fields read from the original Slurm job are optionally
 * propagated depending on MCA component configuration.
 *
 * On success, the resulting SLURM job ID is stored back into the hash table
 * under PRTE_JOB_DATA_JOB_ID.
 *
 * @param[in,out] fields
 *     Hash table containing job configuration inputs and receiving the
 *     resulting job ID on success.
 */
static int prte_ras_slurm_launch_expander_job(pmix_hash_table_t *fields)
{
    if(NULL == fields) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int err = PRTE_SUCCESS;
    int pmix_err = PMIX_SUCCESS;

    char *argv[PRTE_SLURM_MAX_SBATCH_ARGS+1] = {NULL};
    int argc = 0;

    bool have_mem_per_cpu = false;

    char job_id[PRTE_SLURM_JOB_ID_MAX_LEN+1] = {0};
    char *job_id_dyn = NULL;

    const char * const initial_args[] = {"sbatch",
                                "--wrap=sleep infinity", 
                                "--parsable",
                                NULL };

    for (int i = 0; initial_args[i] != NULL; i++) {
        if (argc >= PRTE_SLURM_MAX_SBATCH_ARGS ||
            NULL == (argv[argc] = strdup(initial_args[i]))) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
        argc++;
    }
    
    err = prte_ras_slurm_make_sbatch_arg(fields, record_job_data_fields[PRTE_JOB_DATA_NODES], nodes_format, false, &argc, argv);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    if (prte_mca_ras_slurm_component.propagate_account) {
        err = prte_ras_slurm_make_sbatch_arg(fields, str_fields[STR_ACCOUNT], account_format, false, &argc, argv);

        /* Tolerate not found errors */
        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    if (prte_mca_ras_slurm_component.propagate_partition) {
        err = prte_ras_slurm_make_sbatch_arg(fields, str_fields[STR_PARTITION], partition_format, false, &argc, argv);

        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    if (prte_mca_ras_slurm_component.propagate_qos) {
        err = prte_ras_slurm_make_sbatch_arg(fields, str_fields[STR_QOS], qos_format, false, &argc, argv);

        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

    }

    if (prte_mca_ras_slurm_component.propagate_cwd) {
        err = prte_ras_slurm_make_sbatch_arg(fields, str_fields[STR_CWD], cwd_format, false, &argc, argv);

        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    if(prte_mca_ras_slurm_component.propagate_mem_per_cpu) {
        err = prte_ras_slurm_make_sbatch_arg(fields, num_obj_fields[NUM_OBJ_MEMORY_PER_CPU], 
                                            mem_per_cpu_format, true, &argc, argv);

        if(PRTE_SUCCESS == err) {
            have_mem_per_cpu = true;
        }
        else if(PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    /* Mem per node; only if mem per CPU not already set */
    if(!have_mem_per_cpu && prte_mca_ras_slurm_component.propagate_mem_per_node) {
        err = prte_ras_slurm_make_sbatch_arg(fields, num_obj_fields[NUM_OBJ_MEMORY_PER_NODE], 
                                            mem_per_node_format, true, &argc, argv);

        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    if(prte_mca_ras_slurm_component.propagate_time) {

        err = prte_ras_slurm_make_sbatch_arg(fields, num_obj_fields[NUM_OBJ_TIME_LIMIT], 
                                            time_format, true, &argc, argv);

        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    if(prte_mca_ras_slurm_component.propagate_threads_per_core) {

        err = prte_ras_slurm_make_sbatch_arg(fields, num_obj_fields[NUM_OBJ_THREADS_PER_CORE], 
                                            threads_per_core_format, true, &argc, argv);

        if(PRTE_SUCCESS != err && PRTE_ERR_NOT_FOUND != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    err = prte_ras_slurm_exec_sbatch(argv, job_id);

    if(PRTE_SUCCESS != err) {
        goto cleanup;
    }

    PMIX_OUTPUT_VERBOSE((10, prte_ras_base_framework.framework_output,
                "%s ras:slurm:launch_expander_job: got job ID %s",
                PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), job_id));

    job_id_dyn = strdup(job_id);

    if(NULL == job_id_dyn) {
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(PRTE_ERR_OUT_OF_RESOURCE);
        goto cleanup;
    }

    pmix_err = pmix_hash_table_set_value_ptr(fields, record_job_data_fields[PRTE_JOB_DATA_JOB_ID],
                        strlen(record_job_data_fields[PRTE_JOB_DATA_JOB_ID]), (void*)job_id_dyn);

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_status(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* Now owned by the table */
    job_id_dyn = NULL;

    cleanup:

    if(PRTE_SUCCESS != err && job_id[0] != '\0') {
        /* Prevent hanging resources if failed */
        prte_ras_slurm_kill_job(job_id, NULL);
    }

    if(NULL != job_id_dyn) {
        free(job_id_dyn);
    }

    for(int i = 0; i<PRTE_SLURM_MAX_SBATCH_ARGS+1 && NULL != argv[i]; i++) {
        free(argv[i]);
    }

    return err;
}

/*
 * Run sbatch and capture the submitted Slurm job ID.
 *
 * Executes the command specified by argv in a child process, captures the
 * child's standard output through a pipe, and extracts the leading decimal job
 * ID from that output. The child is then waited on and the result is validated.
 *
 * The function expects output compatible with Slurm's --parsable mode, such
 * as "12345" or "12345;cluster". Only the leading numeric job ID is
 * stored in job_id.
 *
 * @param[in] argv NULL-terminated argument vector for execvp().
 * @param[out] job_id Buffer of size PRTE_SLURM_JOB_ID_MAX_LEN that receives
 *                    the null-terminated numeric job ID on success.
 */
static int prte_ras_slurm_exec_sbatch(char * const *argv, char *job_id) {
    
    if(NULL == argv || NULL == argv[0] || NULL == job_id) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int err = PRTE_SUCCESS;

    job_id[0] = '\0';

    int status;

    size_t n = 0;

    bool overflow = false;

    bool pipe_draining = false;
    bool pipe_drained = false;

    pid_t pid;

    int pipefd[2] = {-1, -1};
    int pipe_err = pipe(pipefd);

    if(pipe_err < 0) {
        err = PRTE_ERR_IN_ERRNO;
        PRTE_ERROR_LOG(err);
        char *strerr = strerror(errno);
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                        "%s ras:slurm:exec_sbatch: pipe failed: %s",
                        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), strerr));
        goto cleanup;   
    }

    pid = fork();

    if(pid < 0) {
        err = PRTE_ERR_IN_ERRNO;
        PRTE_ERROR_LOG(err);
        char *strerr = strerror(errno);
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                        "%s ras:slurm:exec_sbatch: fork failed: %s",
                        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), strerr));
        goto cleanup;
    }

    if (pid == 0) {

        /* Child writes; close read end */
        close(pipefd[0]);
        pipefd[0] = -1;

        /* Redirect output to the pipe */
        if (dup2(pipefd[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }

        /* No longer needed */
        close(pipefd[1]);
        pipefd[1] = -1;

        execvp(argv[0], argv);
        
        /* Something went wrong if we reached this point */
        _exit(127);
    }

    /* Parent reads; close write end */
    close(pipefd[1]);
    pipefd[1] = -1;

    /* Try to get job ID from pipe and drain it after */
    while(!pipe_drained) {
        char c;
        ssize_t r = read(pipefd[0], &c, 1);

        if(1 == r && !pipe_draining)
        {
            /* Slurm job ID, exclusively from digits 0-9 */
            if((n + 1 < PRTE_SLURM_JOB_ID_MAX_LEN) 
                && ('0' <= c && c <= '9')) {
                job_id[n++] = c;
            }

            /* Saw more digits, but had no space for them */
            else if('0' <= c && c <= '9') {
                overflow = true;
                pipe_draining = true;
            }

            /* Ignore initial whitespace */
            else if(!(0 == n && isspace((unsigned char)c))) {
                pipe_draining = true;
            }
        }
        
        /* Nothing more to read */
        else if(0 == r) {
            pipe_drained = true;
        }

        /* Tolerate interruptions */
        else if (r < 0 && errno == EINTR) {
            continue;
        } 

        /* Something went wrong */
        else {
            char *strerr = strerror(errno);
            err = PRTE_ERR_PIPE_READ_FAILURE;
            PRTE_ERROR_LOG(err);
            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:exec_sbatch: pipe read failed: %s",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), strerr));
            break; /* Continue execution to wait for child */
        }
    }

    close(pipefd[0]);
    pipefd[0] = -1;

    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) {
            continue; 
        }

        char *strerr = strerror(errno);
        err = PRTE_ERR_IN_ERRNO;
        PRTE_ERROR_LOG(err);
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
        "%s ras:slurm:exec_sbatch: waitpid failed: %s",
        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), strerr));
        goto cleanup;
    }

    /* Pipe read failed earlier */
    if(PRTE_SUCCESS != err) {
        goto cleanup;
    }

    if (!WIFEXITED(status) || 0 != WEXITSTATUS(status)) {
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
        "%s ras:slurm:exec_sbatch: sbatch failed or exited non-zero",
        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        err = PRTE_ERR_SLURM_SUBMIT_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    if(n == 0 || overflow) {
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
        "%s ras:slurm:exec_sbatch: sbatch exited normally, but got unexpected/truncated output",
        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        err = PRTE_ERR_SLURM_SUBMIT_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    job_id[n] = '\0';

    cleanup:

    if(pipefd[0] >= 0) {
        close(pipefd[0]);
    }

    if(pipefd[1] >= 0) {
        close(pipefd[1]);
    }

    return err;
}

/*
 * Validate that a Slurm job ID is valid according to expected syntax
 *
 * A valid Slurm job ID must be non-NULL, non-empty, must not exceed
 * PRTE_SLURM_JOB_ID_MAX_LEN characters, and must contain only decimal digits.
 * 
 * @param[in] slurm_jobid  Null-terminated Slurm job ID string to validate.
 */
static int prte_ras_slurm_validate_jobid(const char *slurm_jobid) {

    if (NULL == slurm_jobid) {
        return PRTE_ERR_BAD_PARAM;
    }

    size_t id_len = strnlen(slurm_jobid, PRTE_SLURM_JOB_ID_MAX_LEN+1);
    if (0 == id_len || id_len > PRTE_SLURM_JOB_ID_MAX_LEN) {
        return PRTE_ERR_BAD_PARAM;
    }

    for (size_t i = 0; i < id_len; ++i) {
        if (!isdigit((unsigned char)slurm_jobid[i])) {
            return PRTE_ERR_BAD_PARAM;
        }
    }

    return PRTE_SUCCESS;
}

/*
 * Cancel a Slurm job using scancel.
 *
 * If scancel returns an error, the first line of stderr/stdout output is copied
 * into err_msg. On success, err_msg is cleared.
 *
 * @param[in]  slurm_jobid  Null-terminated Slurm job ID string to cancel.
 * @param[out] err_msg      Writable buffer of size PRTE_SLURM_ERR_STR_MAX_SIZE
 *                          for Slurm output on failure, or NULL if not required.
 */
static int prte_ras_slurm_kill_job(const char *slurm_jobid, char *err_msg) {

    if(NULL == slurm_jobid) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    if(NULL != err_msg) {
        err_msg[0] = '\0';
    }

    int err = PRTE_SUCCESS;

    /* Make sure the job ID given is something reasonable */
    err = prte_ras_slurm_validate_jobid(slurm_jobid);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        return err;
    }

    static const char *cmd_format = "scancel %s 2>&1";

    char *cmd = NULL;

    FILE *fp = NULL;

    if(0 > asprintf(&cmd, cmd_format, slurm_jobid)) {
        cmd = NULL;
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    fp = popen(cmd, "r");

    if(NULL == fp) {
        err = PRTE_ERR_FILE_OPEN_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    if(NULL != err_msg) {
        char *buf = fgets(err_msg, PRTE_SLURM_ERR_STR_MAX_SIZE, fp);

        /* Copy output output into provided memory, truncating if necessary */
        if(NULL != buf) {
            size_t len = strcspn(buf, "\n");

            if (buf[len] == '\n') {
                buf[len] = '\0';
            } else if (len == PRTE_SLURM_ERR_STR_MAX_SIZE - 1) {
                memcpy(buf + PRTE_SLURM_ERR_STR_MAX_SIZE - 4, "...", 3);
            }
        } 
    }

    int status = pclose(fp);
    fp = NULL;

    if (-1 == status) {
        err = PRTE_ERR_FILE_OPEN_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    if (!WIFEXITED(status) || 0 != WEXITSTATUS(status)) {
        err = PRTE_ERR_SLURM_CANCEL_FAILURE;
        goto cleanup;
    }

    cleanup:

    if(NULL != err_msg && PRTE_ERR_SLURM_CANCEL_FAILURE != err) {
        err_msg[0] = '\0';
    }

    if(NULL != fp) {
        pclose(fp);
    }

    free(cmd);

    return err;
}

int prte_ras_slurm_assign_new_session(const char *slurm_jobid, const char *alloc_refid) {
    
    if(NULL == slurm_jobid) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int err = PRTE_SUCCESS;

    err = prte_ras_slurm_validate_jobid(slurm_jobid);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        return err;
    }
    
    const int base = 10;
    char *end = NULL;

    uint32_t slurm_id_uint;
    unsigned long slurm_id_ulong;
    
    errno = 0;
    id_ulong = strtoul(slurm_jobid, &end, base);

    if (end == slurm_jobid || '\0' != end
     || errno == ERANGE || id_ulong > UINT32_MAX) {
        err = PRTE_ERR_BAD_PARAM;
        PRTE_ERROR_LOG(err);
        return err;
    }

    id_uint = (uint32_t)id_ulong;

    char *alloc_refid_dup = NULL;

    if(NULL != alloc_refid) {
        alloc_refid_dup = strdup(alloc_refid);
        if(NULL == alloc_refid_dup) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            PRTE_ERROR_LOG(err);
            return err;
        }
    }

    prte_session_t *session = NULL;

    session = PMIX_NEW(prte_session_t);

    session->session_id = slurm_id_uint;

    if(NULL != alloc_refid_dup) {
        session->alloc_refid = alloc_refid_dup;
    }

    PMIX_LIST_FOREACH(node, nodes, prte_node_t) {
        PMIX_RETAIN(node);

        int idx = pmix_pointer_array_add(session->nodes, nd);
        if (0 > idx) {
            PMIX_RELEASE(nd);
            PMIX_RELEASE(session);
            return PRTE_ERR_OUT_OF_RESOURCE;
        }
    }

    err = prte_set_session_object(session);
    if (PRTE_SUCCESS != err) {
        PMIX_RELEASE(session);
        PRTE_ERROR_LOG(err);
        free(alloc_refid_dup);
        return err;
    }

    /* Now owned by the session */
    alloc_refid_dup = NULL;


}