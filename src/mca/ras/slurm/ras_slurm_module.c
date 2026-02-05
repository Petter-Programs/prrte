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
static int prte_ras_slurm_find_next_quoted(const char **str_in, const char **start_quote, const char **end_quote);
static int prte_ras_slurm_find_next_delimited_obj(const char **str_in, char open_ch, char close_ch, const char **start_obj, const char **end_obj);
static int prte_ras_slurm_strip_json_whitespace(char *json_text);
static int prte_ras_slurm_skip_json_value(const char **line);
static int prte_ras_slurm_string_is_safe(const char *s, bool *safe);
static int prte_ras_slurm_match_json_entries(pmix_hash_table_t* type_table, pmix_hash_table_t* val_table, char *slurm_json);
static int prte_ras_slurm_wipe_dyn_hashtable(pmix_hash_table_t *table);
static int prte_ras_slurm_extract_job_fields(pmix_hash_table_t *values_table);
static int prte_ras_slurm_make_job_field(pmix_hash_table_t *fields, const char *field_name, const char *field_format, char **field_out, bool obj_num);
static int prte_ras_slurm_launch_expander_job(pmix_hash_table_t *fields);

/* Slurm sbatch formats */
static const char *account_format   = "#SBATCH --account=%s\n";
static const char *partition_format = "#SBATCH --partition=%s\n";
static const char *qos_format       = "#SBATCH --qos=%s\n";
static const char *cwd_format       = "#SBATCH --chdir=%s\n";
static const char *mem_per_cpu_format  = "#SBATCH --mem-per-cpu=%ld\n";
static const char *mem_per_node_format = "#SBATCH --mem=%ld\n";
static const char *time_format = "#SBATCH --time=%s\n";
static const char *nodes_format = "#SBATCH --nodes=%s\n";

/* States for parsing Slurm JSON output */
typedef enum {
    STATE_KEY,
    STATE_COLON,
    STATE_VAL,
    STATE_SET,
} TextParseState;

/* Markers for expected types when parsing Slurm JSON */
static char const * const str_marker = "str"; /* text-based entry */
static char const * const num_marker = "num"; /* entry only with digits 0 to 9 */
static char const * const bool_marker = "bool"; /* entry that is true or false */
static char const * const obj_marker = "obj";  /* JSON object inside brackets {} */
static char const * const arr_obj_marker = "arr_obj"; /* Array containing single JSON object */

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
    NUM_OBJ_FIELD_COUNT
};

static const char *const num_obj_fields[NUM_OBJ_FIELD_COUNT] = {
    [NUM_OBJ_MEMORY_PER_CPU]  = "memory_per_cpu",
    [NUM_OBJ_MEMORY_PER_NODE] = "memory_per_node",
    [NUM_OBJ_TIME_LIMIT] = "time_limit",
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

static const char *const num_obj_subfield_types[] = {
    bool_marker,
    bool_marker,
    num_marker,
};

enum slurm_prte_request_fields {
    PRTE_REQUEST_NODES,
    PRTE_REQUEST_COUNT
};

static const char *const prte_request_fields[PRTE_REQUEST_COUNT] = {
    [PRTE_REQUEST_NODES]      = "nodes",
};

static const size_t total_fields_len = STR_FIELD_COUNT + NUM_OBJ_FIELD_COUNT + PRTE_REQUEST_COUNT;

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

static void modify(prte_pmix_server_req_t *req)
{
    int prte_err = PRTE_SUCCESS;
    int pmix_err = PMIX_SUCCESS;

    pmix_hash_table_t slurm_jobfields;
    bool have_slurm_jobfields = false;

    char *nodes_string = NULL;

    if(PMIX_ALLOC_EXTEND == req->allocdir) {
     
        PMIX_CONSTRUCT(&slurm_jobfields, pmix_hash_table_t);

        pmix_err = pmix_hash_table_init(&slurm_jobfields, total_fields_len);

        if(PMIX_SUCCESS != pmix_err) {
            goto cleanup;
        }

        prte_err = prte_ras_slurm_extract_job_fields(&slurm_jobfields);

        if(PRTE_SUCCESS != prte_err) {

            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                                "%s ras:slurm:modify: failed to parse fields from current Slurm job",
                                PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));

            goto cleanup;
        }

        uint32_t num_nodes;
        bool found = false;

        for (size_t i = 0; i < req->ninfo; ++i) {

            if (0 == strcmp(req->info[i].key, PMIX_NUM_NODES)) {

                if (req->info[i].value.type != PMIX_UINT32) {
                    prte_err = PRTE_ERR_BAD_PARAM;
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

            prte_err = PRTE_ERR_NOT_FOUND;
            goto cleanup;
        }

        int rc = asprintf(&nodes_string, "%" PRIu32, num_nodes);
        
        if(0 > rc) {
            prte_err = PRTE_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }

        pmix_err = pmix_hash_table_set_value_ptr(&slurm_jobfields, prte_request_fields[PRTE_REQUEST_NODES],
                                strlen(prte_request_fields[PRTE_REQUEST_NODES]), (void*)nodes_string);

        if(PMIX_SUCCESS != pmix_err) {
            goto cleanup;
        }

        prte_err = prte_ras_slurm_launch_expander_job(&slurm_jobfields);

        if(PRTE_SUCCESS == prte_err) {
            printf("Yippie, launched an expander job!\n");
        }
    }

    cleanup:

    free(nodes_string);

    if(PMIX_SUCCESS != pmix_err) {
        prte_err = prte_pmix_convert_rc(pmix_err);
    }

    if(prte_err != PRTE_SUCCESS) {
        PRTE_ERROR_LOG(prte_err);
    }

    if(have_slurm_jobfields) {
        prte_ras_slurm_wipe_dyn_hashtable(&slurm_jobfields);

        PMIX_DESTRUCT(&slurm_jobfields);
    }

    req->status = PMIX_ERR_NOT_SUPPORTED;
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
 * Parse a null-terminated input text and find the first entry
 * deliminated by a pair of unescaped quotes inside it. Advance
 * the input string address one step after the end quote.
 * 
 * @param **str_in       Address of a `const char *` pointing to the input text, which will be advanced.
 * @param **start_quote  Address of a `const char *` that will receive the start quote address.
 * @param **end_quote    Address of a `const char *` that will receive the end quote address.
 */
static int prte_ras_slurm_find_next_quoted(const char **str_in, const char **start_quote, const char **end_quote)
{
    if(!str_in || !*str_in || !start_quote || !end_quote) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    bool start_quote_found = false;
    int backslash_count = 0;

    while('\0' != **str_in) {
        /* Even number of backslashes --> unescaped */
        if('\"' == **str_in && 0 == backslash_count % 2) {
            if(!start_quote_found) {
                *start_quote = *str_in;
                start_quote_found = true;
                backslash_count = 0;
            }
            else
            {
                *end_quote = *str_in;
                (*str_in)++;
                return PRTE_SUCCESS;
            }
        } else if ('\\' == **str_in) {
            backslash_count++;
        } else {
            backslash_count = 0;
        }
        (*str_in)++;
    }

    return PRTE_ERR_NOT_FOUND;
}

/*
 * Parse a null-terminated string and locate the next object delimited
 * by balanced open/close characters. Delimiters appearing inside
 * quoted strings are ignored.
 *
 * On success, the input pointer is advanced to the character
 * immediately following the closing delimiter.
 *
 * @param str_in      Address of a `const char *` input pointer, advanced as parsed.
 * @param open_ch     Opening delimiter character (e.g. '{', '[', '(').
 * @param close_ch    Closing delimiter character (e.g. '}', ']', ')').
 * @param start_obj   Receives the address of the opening delimiter.
 * @param end_obj     Receives the address of the closing delimiter.
 *
 * @return PRTE_SUCCESS on success, or a PRTE error code on failure.
 */
static int prte_ras_slurm_find_next_delimited_obj(const char **str_in, char open_ch, char close_ch, const char **start_obj, const char **end_obj)
{
    if (!str_in || !*str_in || !start_obj || !end_obj) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int bracket_counter = 0;

    const char *next_quote_start = NULL;
    const char *next_quote_end = NULL;

    while ('\0' != **str_in) {

        /* ignore any quoted area for counting delimiters */
        if ('"' == **str_in) {
            int res = prte_ras_slurm_find_next_quoted(
                str_in, &next_quote_start, &next_quote_end);
            if (PRTE_SUCCESS != res) {
                PRTE_ERROR_LOG(res);
                return res;
            }
            
            /* cursor already advanced by prte_ras_slurm_find_next_quoted */
            continue;
        }

        else if (open_ch == **str_in) {
            if (0 == bracket_counter) {
                *start_obj = *str_in;
            }
            bracket_counter++;
        }

        else if (close_ch == **str_in) {
            bracket_counter--;

        if('{' == **str_in || '}' == **str_in)
            if (0 > bracket_counter) {
                PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
                return PRTE_ERR_BAD_PARAM;
            }

            if (0 == bracket_counter) {
                *end_obj = *str_in;
                (*str_in)++;
                return PRTE_SUCCESS;
            }
        }

        (*str_in)++;
    }

    /* either none found or incorrectly formatted */
    return PRTE_ERR_NOT_FOUND;
}

/*
 * Strip JSON whitespace characters from a string, ignoring quoted regions.
 * The operation is performed in-place.
 *
 * @param json_text   Mutable null-terminated JSON text buffer.
 */
static int prte_ras_slurm_strip_json_whitespace(char *json_text)
{
    if (NULL == json_text) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    const char *read_ptr = json_text;
    char *write_ptr = json_text;

    const char *quote_start = NULL;
    const char *quote_end = NULL;

    while ('\0' != *read_ptr) {

        /* copy quoted regions verbatim */
        if ('"' == *read_ptr) {
            const char *quote_read = read_ptr;
            int res = prte_ras_slurm_find_next_quoted(
                &quote_read, &quote_start, &quote_end);
            if (PRTE_SUCCESS != res) {
                PRTE_ERROR_LOG(res);
                return res;
            }

            while (read_ptr < quote_read) {
                *write_ptr++ = *read_ptr++;
            }
            continue;
        }

        /* outside quotes: skip whitespace */
        if (isspace((unsigned char)*read_ptr)) {
            read_ptr++;
            continue;
        }

        *write_ptr++ = *read_ptr++;
    }

    *write_ptr = '\0';
    return PRTE_SUCCESS;
}

/*
 * Skip a single JSON value and advance the parse cursor
 *
 * Skips over the JSON value starting at *line and advances the cursor to
 * the first character following the value. Supports strings, objects,
 * arrays, and primitive values
 *
 * @param line A pointer to the current position in a JSON string; updated
 *             to point past the skipped value
 */
static int prte_ras_slurm_skip_json_value(const char **line) {
    const char *start = *line;
    const char *end = NULL;
    int err;

    switch (**line) {
    case '"':
        err = prte_ras_slurm_find_next_quoted(line, &start, &end);
        if (PRTE_SUCCESS != err) {
            return err;
        } 
        return PRTE_SUCCESS;

    case '{':
        err = prte_ras_slurm_find_next_delimited_obj(
            line, '{', '}', &start, &end);
        if (PRTE_SUCCESS != err) {
            return err;
        } 
        return PRTE_SUCCESS;

    case '[':
        err = prte_ras_slurm_find_next_delimited_obj(
            line, '[', ']', &start, &end);
        if (PRTE_SUCCESS != err) {
            return err;
        }
        return PRTE_SUCCESS;

    default:
        /* number, true, false, null */
        while (**line &&
               **line != ',' &&
               **line != '}' &&
               **line != ']') {
            (*line)++;
        }
        return PRTE_SUCCESS;
    }
}

/**
 * @brief Check whether a string contains characters that increase attack surface.
 *
 * Rejects the string if it contains any ASCII control characters, space,
 * or characters that may trigger interpretation by downstream parsers
 */
static int prte_ras_slurm_string_is_safe(const char *s, bool *safe)
{
    if (NULL == s || NULL == safe) {
        return PRTE_ERR_BAD_PARAM;
    }

    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        unsigned char c = *p;

        /* reject control characters */
        if (c < 0x20 || c == 0x7f) {
            *safe = false;
        }

        /* reject interpreter / metacharacters and space */
        switch (c) {
            case '"': case '\'': case '\\': case '`':
            case '$': case ';': case '&':  case '|':
            case '<': case '>': case '(': case ')':
            case '{': case '}': case '[': case ']':
            case '*': case '?': case '!': case ' ':
                *safe = false;
            default:
                break;
        }
    }

    return PRTE_SUCCESS;
}

/*
 * Parse a Slurm JSON-format input and extract a fixed set of expected keys
 *
 * Performs a constrained JSON parse intended for SLURM-generated JSON with
 * a known schema. Only keys present in the type table are considered.
 * Parsing fails if any expected key is missing or malformed. The input
 * string is modified in-place during parsing.
 *
 * On success, values stored in the value table are heap-allocated and owned
 * by the table. On error, any values inserted by this function are removed
 * and freed.
 *
 * @param type_table Hash table mapping expected JSON keys to type markers
 * @param val_table  Hash table to receive extracted key/value pairs
 * @param slurm_json NUL-terminated JSON string; modified during parsing
 */
static int prte_ras_slurm_match_json_entries(pmix_hash_table_t* type_table, pmix_hash_table_t* val_table, char *slurm_json)
{
    if(!slurm_json || !type_table || !val_table) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    int count_found = 0;
    size_t expected_count = pmix_hash_table_get_size(type_table);

    int pmix_err = PMIX_SUCCESS;
    int err = PRTE_SUCCESS;

    /* helper to indicate what stage of parsing we're in */
    TextParseState parse_state = STATE_KEY;

    const char *open_quote = NULL;
    const char *close_quote = NULL;

    char *key_ptr = NULL;
    char *val_ptr = NULL;

    /* remove any whitespace or linebreaks, as JSON does too */
    prte_ras_slurm_strip_json_whitespace(slurm_json);

    const char *json_cursor = slurm_json;

    while('\0' != *json_cursor && expected_count > count_found || STATE_SET == parse_state) {

        switch(parse_state) {
            
            /* look for keys in unescaped quotes */
            case STATE_KEY: {

                if(PRTE_SUCCESS == prte_ras_slurm_find_next_quoted(&json_cursor, &open_quote, &close_quote)) {

                    const char * start_text = open_quote+1;
                    size_t len = close_quote - start_text;

                    key_ptr = strndup(start_text, len);

                    if(NULL == key_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        goto cleanup;
                    }

                    parse_state = STATE_COLON;
                    break;
                
                }
                else {
                    err = PRTE_ERR_NOT_FOUND;
                    goto cleanup;
                }
            }

            /* look for colon following a potential key */
            case STATE_COLON: {
                if(*json_cursor == ':') {
                    parse_state = STATE_VAL;
                } else {
                    parse_state = STATE_KEY;
                }
                json_cursor++;
                break;
            }

            /* we have a key; check if it is of interest and capture its value */
            case STATE_VAL: {

                /* must not be NULL */
                size_t key_len = strlen(key_ptr);
                
                char *type_marker;

                /* key not in hash table */ 
                if (PMIX_SUCCESS != pmix_hash_table_get_value_ptr(type_table, key_ptr,
                                        key_len, (void**)&type_marker)) {
                        free(key_ptr);
                        key_ptr = NULL;

                        err = prte_ras_slurm_skip_json_value(&json_cursor);
                        if (PRTE_SUCCESS != err) {
                            PRTE_ERROR_LOG(err);
                            goto cleanup;
                        }

                        parse_state = STATE_KEY;
                        break;
                }

                /* string key */
                if(type_marker == str_marker) {

                    /* expected a string and only a string */
                    if(*json_cursor != '"') {
                        err = PRTE_ERR_NOT_FOUND;
                        PRTE_ERROR_LOG(err);
                        goto cleanup; 
                    }

                    err = prte_ras_slurm_find_next_quoted(&json_cursor, &open_quote, &close_quote);
        
                    if(PRTE_SUCCESS != err) {
                        PRTE_ERROR_LOG(err);
                        goto cleanup; 
                    }

                    const char * start_text = open_quote+1;
                    size_t len = close_quote-start_text;

                    /* maximum size limit for any field */
                    if(len > 4096) {
                        err = PRTE_ERR_BAD_PARAM;
                        goto cleanup;
                    }

                    val_ptr = strndup(start_text, len);

                    if(NULL == val_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    bool safe = true;

                    /* filter out some potentially malicious characters */
                    if (PRTE_SUCCESS != prte_ras_slurm_string_is_safe(val_ptr, &safe) || !safe) {
                        err = PRTE_ERR_BAD_PARAM;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    parse_state = STATE_SET;
                }
                
                /* numeric key */
                else if(type_marker == num_marker) {
                    const char *line_start = json_cursor;
                    size_t len = 0;

                    /* might need to be loosened in the future, but OK restriction for now */
                    while (isdigit((unsigned char)*json_cursor)) {
                        len++;
                        json_cursor++;
                    }

                    /* started with some unexpected character */
                    if(0 == len) {
                        err = PRTE_ERR_NOT_FOUND;
                        PRTE_ERROR_LOG(err);
                        goto cleanup; 
                    }
                    
                    val_ptr = strndup(line_start, len);

                    if(NULL == val_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    parse_state = STATE_SET;
                }

                /* boolean type */
                else if(type_marker == bool_marker) {
                    
                    /* technically, a value like trueX would be accepted here,
                    * but the goal is not a fully compliant parser; we expect
                    * the Slurm output to be well behaved */

                    char *val;
                    
                    if(0 == strncmp(json_cursor, "true", 4)) {
                        val = "true";
                        json_cursor+=4;
                    }

                    else if(0 == strncmp(json_cursor, "false", 5)) {
                        val = "false";
                        json_cursor+=5;
                    }

                    else {
                        err = PRTE_ERR_NOT_FOUND;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }
                    
                    val_ptr = strdup(val);

                    if(NULL == val_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    parse_state = STATE_SET;
                }

                /* JSON object key */
                else if(type_marker == obj_marker) {
                    
                    if(*json_cursor != '{') {
                        err = PRTE_ERR_NOT_FOUND;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    const char *obj_start = NULL;
                    const char *obj_end = NULL;
                    
                    err = prte_ras_slurm_find_next_delimited_obj(&json_cursor, '{', '}', &obj_start, &obj_end);

                    if(PRTE_SUCCESS != err) {
                        PRTE_ERROR_LOG(err);
                        goto cleanup;   
                    }

                    const char * start_content = obj_start+1;
                    size_t len = obj_end - start_content;

                    val_ptr = strndup(start_content, len);

                    if(NULL == val_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }
                    
                    parse_state = STATE_SET;
                }

                /* array containing a single JSON object */
                else if(type_marker == arr_obj_marker) {

                    if(*json_cursor != '[') {
                        err = PRTE_ERR_NOT_FOUND;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    const char *arr_start = NULL;
                    const char *arr_end = NULL;
                    
                    err = prte_ras_slurm_find_next_delimited_obj(&json_cursor, '[', ']', &arr_start, &arr_end);

                    if(PRTE_SUCCESS != err) {
                        PRTE_ERROR_LOG(err);
                        goto cleanup;   
                    }

                    json_cursor = arr_start+1;

                    const char *obj_start = NULL;
                    const char *obj_end = NULL;

                    err = prte_ras_slurm_find_next_delimited_obj(&json_cursor, '{', '}', &obj_start, &obj_end);

                    if(PRTE_SUCCESS != err) {
                        PRTE_ERROR_LOG(err);
                        goto cleanup;   
                    }

                    /* we expect this structure [{object1}] when stripped of whitespace */
                    if(obj_start != arr_start+1 || arr_end != obj_end+1) {
                        err = PRTE_ERR_NOT_FOUND;
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    const char * start_content = obj_start+1;
                    size_t len = obj_end - start_content;

                    val_ptr = strndup(start_content, len);

                    if(NULL == val_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        goto cleanup;
                    }

                    parse_state = STATE_SET;
                }

                /* unknown or unexpected type */ 
                else {
                    err = PRTE_ERR_BAD_PARAM;
                    PRTE_ERROR_LOG(err);
                    goto cleanup;
                }

                break;
            }

            /* NOTE: json_cursor may be out of bounds here !! */
            case STATE_SET: {

                void *existing;

                /* check for duplicate keys */
                if (PMIX_SUCCESS != pmix_hash_table_get_value_ptr(val_table, key_ptr,
                                         strlen(key_ptr), (void**)&existing)) {
                    
                    pmix_err = pmix_hash_table_set_value_ptr(val_table, key_ptr, strlen(key_ptr), (void*)val_ptr);

                    if(PMIX_SUCCESS != pmix_err) {
                        err = prte_pmix_convert_rc(pmix_err);
                        PRTE_ERROR_LOG(err);
                        goto cleanup;
                    }

                    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                    "%s ras:slurm:match_json_entries: found match for %s",
                    PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), key_ptr));
                    
                    /* do not free as pointer is in in hash table */
                    val_ptr = NULL;

                    free(key_ptr);
                    key_ptr = NULL;

                    /* extracted the value */
                    count_found++;

                } else {
                    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                    "%s ras:slurm:match_json_entries: ignoring duplicate key %s",
                    PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), key_ptr));
                    /* duplicates */
                    free(key_ptr);
                    free(val_ptr);
                    key_ptr = NULL;
                    val_ptr = NULL;
                }
                
                parse_state = STATE_KEY;

                break;
            }

        }
    }

    if(expected_count != count_found) {
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:match_json_entries: found %d entries but expected %d",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), count_found, expected_count));
        err = PRTE_ERR_NOT_FOUND;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    cleanup:

    free(key_ptr);
    free(val_ptr);

    return err;
}

/**
 * @brief Wipe and free() all values in a PMIx hash table.
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

    int pmix_err;

    pmix_err = pmix_hash_table_get_first_key_ptr(table, &key, &key_size, &val, &node);

    if(PMIX_SUCCESS != pmix_err) {
        return prte_pmix_convert_rc(pmix_err);
    }
    
    free(val);
    prev_node = node;

    while (PMIX_SUCCESS ==
        pmix_hash_table_get_next_key_ptr(table, &key, &key_size, &val, &prev_node, &node)) {
        free(val);
        prev_node = node;
    }

    pmix_err = pmix_hash_table_remove_all(table);
    return prte_pmix_convert_rc(pmix_err);
}

static int prte_ras_slurm_extract_job_fields(pmix_hash_table_t *values_table)
{
    int pmix_err = PMIX_SUCCESS;
    int err = PRTE_SUCCESS;

    FILE *fp = NULL;

    char *slurm_json = NULL;
    char *jobs_json_obj = NULL;
    char *cmd = NULL;

    char *slurm_jobid;
    if (NULL == (slurm_jobid = getenv("SLURM_JOBID"))) {
        PRTE_ERROR_LOG(PRTE_ERR_NOT_FOUND);
        return PRTE_ERR_NOT_FOUND;
    }

    /* make sure the job ID read is something reasonable */
    
    size_t id_len = strlen(slurm_jobid);
    if (id_len == 0 || id_len > 20) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    for (const char *p = slurm_jobid; *p != '\0'; ++p) {
        if (!isdigit((unsigned char)*p)) {
            PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
            return PRTE_ERR_BAD_PARAM;
        }
    }

    pmix_hash_table_t tmp_table;
    pmix_hash_table_t types_table;

    PMIX_CONSTRUCT(&tmp_table, pmix_hash_table_t);
    PMIX_CONSTRUCT(&types_table, pmix_hash_table_t);

    /*
    * use hash tables to match expected keys with expected value types
    * and to store their extracted result 
    */

    pmix_err = pmix_hash_table_init(&types_table, total_fields_len);

    if(PMIX_SUCCESS != pmix_err) {
        PMIX_DESTRUCT(&types_table);
        return prte_pmix_convert_rc(pmix_err);
    }

    pmix_err = pmix_hash_table_init(&tmp_table, total_fields_len);

    if(PMIX_SUCCESS != pmix_err) {
        PMIX_DESTRUCT(&types_table);
        PMIX_DESTRUCT(&tmp_table);
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        return err;
    }

    /* TODO: how about Slurm-side errors */

    char *cmd_format = "scontrol show job %s --json";

    if(0 > asprintf(&cmd, cmd_format, slurm_jobid)) {
        cmd = NULL;
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    fp = popen(cmd, "r");

    if(!fp) {
        err = PRTE_ERR_FILE_OPEN_FAILURE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    size_t curr_limit = 4096;
    size_t len = 0;

    slurm_json = malloc(curr_limit);

    if (!slurm_json) {
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    char buf[1024];
    while (fgets(buf, sizeof(buf), fp)) {
        size_t chunk_len = strlen(buf);

        if (len + chunk_len + 1 > curr_limit) {
            curr_limit *= 2;

            if(curr_limit > PRTE_SLURM_JOB_INFO_MAX_SIZE) {
                err = PRTE_ERR_MEM_LIMIT_EXCEEDED;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            char *tmp = realloc(slurm_json, curr_limit);
            if (!tmp) {
                err = PRTE_ERR_OUT_OF_RESOURCE;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            slurm_json = tmp;
        }

        memcpy(slurm_json + len, buf, chunk_len);
        len += chunk_len;
    }

    slurm_json[len] = '\0';

    pclose(fp);
    fp = NULL;

    if(0 == strlen(slurm_json)) {
        err = PRTE_ERR_NOT_FOUND;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* first, extract the object inside the "jobs" field of the Slurm JSON. 
     * this implicitly also validates that we got exactly one match, because
     * prte_ras_slurm_match_json_entries checks for exactly one JSON object
     * inside an array structure, and would fail if there is more */

    pmix_err = pmix_hash_table_set_value_ptr(&types_table, jobs_field,
                            strlen(jobs_field), (void*)arr_obj_marker);

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    err = prte_ras_slurm_match_json_entries(&types_table, &tmp_table, slurm_json);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    char *jobs_json_obj_val;

    pmix_err = pmix_hash_table_get_value_ptr(&tmp_table, jobs_field,
                                        strlen(jobs_field), (void**)&jobs_json_obj_val);
    
    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }
    
    pmix_err = pmix_hash_table_remove_value_ptr(&types_table, jobs_field, strlen(jobs_field));

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* currently owned by the tmp table, which we will clear later, so duplicate it */
    jobs_json_obj = strdup(jobs_json_obj_val);

    if(NULL == jobs_json_obj) {
        err = PRTE_ERR_OUT_OF_RESOURCE;
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* we've extracted a valid "jobs" section. now extract the complex numeric
     * fields that have "set", "infinite", and "number" fields into a temporary
     * table which we can then process */

    for(size_t i = 0; i < NUM_OBJ_SUBFIELD_COUNT && PMIX_SUCCESS == pmix_err; i++) {
        pmix_err = pmix_hash_table_set_value_ptr(&types_table, num_obj_fields[i],
                                strlen(num_obj_fields[i]), (void*)obj_marker);
    }

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    err = prte_ras_slurm_match_json_entries(&types_table, &tmp_table, jobs_json_obj);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    for(size_t i = 0; i < NUM_OBJ_SUBFIELD_COUNT && PMIX_SUCCESS == pmix_err; i++) {
        pmix_err = pmix_hash_table_remove_value_ptr(&types_table, num_obj_fields[i], strlen(num_obj_fields[i]));
    }

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    /* numbers in JSON object format: set, infinite, number 
     *  here, set the names of these subfields and their types */
    for(size_t i = 0; i < NUM_OBJ_SUBFIELD_COUNT && PMIX_SUCCESS == pmix_err; i++) {
        pmix_err = pmix_hash_table_set_value_ptr(&types_table, num_obj_subfields[i],
                                strlen(num_obj_subfields[i]), (void*)num_obj_subfield_types[i]);
    }

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    for(size_t i = 0; i < NUM_OBJ_SUBFIELD_COUNT && PRTE_SUCCESS == err; i++) {

        char *num_obj_field;

        pmix_err = pmix_hash_table_get_value_ptr(&tmp_table, num_obj_fields[i],
                                        strlen(num_obj_fields[i]), (void**)&num_obj_field);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_rc(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        err = prte_ras_slurm_match_json_entries(&types_table, &tmp_table, num_obj_field);
        
        if(PRTE_SUCCESS != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        char *set;
        char *infinite;
        char *value;

        pmix_err = pmix_hash_table_get_value_ptr(&tmp_table, num_obj_subfields[NUM_OBJ_SUBFIELD_SET],
                                strlen(num_obj_subfields[NUM_OBJ_SUBFIELD_SET]), (void**)&set);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_rc(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        pmix_err = pmix_hash_table_get_value_ptr(&tmp_table, num_obj_subfields[NUM_OBJ_SUBFIELD_INFINITE],
                               strlen(num_obj_subfields[NUM_OBJ_SUBFIELD_INFINITE]), (void**)&infinite);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_rc(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        pmix_err = pmix_hash_table_get_value_ptr(&tmp_table, num_obj_subfields[NUM_OBJ_SUBFIELD_NUMBER],
                                strlen(num_obj_subfields[NUM_OBJ_SUBFIELD_NUMBER]), (void**)&value);

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_rc(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        if(0 == strcmp(set, "false")) {
            char *empty_dyn = strdup(unset_num_marker);

            if(NULL == empty_dyn) {
                err = PRTE_ERR_OUT_OF_RESOURCE;
                goto cleanup;
            }

            pmix_err = pmix_hash_table_set_value_ptr(values_table, num_obj_fields[i],
                        strlen(num_obj_fields[i]), empty_dyn);

            if(PMIX_SUCCESS != pmix_err) {
                err = prte_pmix_convert_rc(pmix_err);
                PRTE_ERROR_LOG(err);
            }
        }

        else if(0 == strcmp(infinite, "true")) {

            char *inf_dyn = strdup(infinite_num_marker);

            if(NULL == inf_dyn) {
                err = PRTE_ERR_OUT_OF_RESOURCE;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            pmix_err = pmix_hash_table_set_value_ptr(values_table, num_obj_fields[i],
            strlen(num_obj_fields[i]), inf_dyn);
        }

        else {
            char *val_dup = strdup(value);

            if(NULL == val_dup) {
                err = PRTE_ERR_OUT_OF_RESOURCE;
                PRTE_ERROR_LOG(err);
                goto cleanup;
            }

            pmix_err = pmix_hash_table_set_value_ptr(values_table, num_obj_fields[i],
            strlen(num_obj_fields[i]), val_dup);
        }

        if(PMIX_SUCCESS != pmix_err) {
            err = prte_pmix_convert_rc(pmix_err);
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }

        /* wipe the temporary values as we no longer need them */
        err = prte_ras_slurm_wipe_dyn_hashtable(&tmp_table);

        if(PRTE_SUCCESS != err) {
            PRTE_ERROR_LOG(err);
            goto cleanup;
        }
    }

    pmix_err = pmix_hash_table_remove_all(&types_table);

    /* finally, find the string and numeric fields and add them to our value table */

    for(size_t i = 0; i < STR_FIELD_COUNT && PMIX_SUCCESS == pmix_err; i++) {
        pmix_err = pmix_hash_table_set_value_ptr(&types_table, str_fields[i],
                              strlen(str_fields[i]), (void*)str_marker);
    }

    if(PMIX_SUCCESS != pmix_err) {
        err = prte_pmix_convert_rc(pmix_err);
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    err = prte_ras_slurm_match_json_entries(&types_table, values_table, jobs_json_obj);

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
        goto cleanup;
    }

    cleanup:

    if(NULL != fp) {
        pclose(fp);
    }

    free(slurm_json);
    free(jobs_json_obj);
    free(cmd);

    prte_ras_slurm_wipe_dyn_hashtable(&tmp_table);

    PMIX_DESTRUCT(&tmp_table);
    PMIX_DESTRUCT(&types_table);

    return err;
}

static int prte_ras_slurm_make_job_field(pmix_hash_table_t *fields, const char *field_name, const char *field_format, char **field_out, bool obj_num) {
    
    char *stored_val;

    int pmix_err = pmix_hash_table_get_value_ptr(fields, field_name,
                        strlen(field_name), (void**)&stored_val);

    if(PMIX_SUCCESS != pmix_err || NULL == stored_val || 0 == strlen(stored_val)) {
        return PRTE_ERR_NOT_FOUND;
    }

    if(obj_num) {
        /* handle both as just unset for now */
        if(0 == strcmp(stored_val, unset_num_marker)
        || 0 == strcmp(stored_val, infinite_num_marker)) {
            field_format = "%s";
            stored_val = "";
        }
    }

    int rc = asprintf(field_out, field_format, stored_val);

    if(0 > rc) {
        *field_out = NULL;
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    return PRTE_SUCCESS;
}

static int prte_ras_slurm_launch_expander_job(pmix_hash_table_t *fields) {

    int prte_err = PRTE_SUCCESS;
    int pmix_err = PMIX_SUCCESS;

    char *account_field = NULL;
    char *partition_field = NULL;
    char *qos_field = NULL;
    char *cwd_field = NULL;
    char *mem_per_cpu_field = NULL;
    char *mem_per_node_field = NULL;
    char *time_limit_field = NULL;
    char *nodes_field = NULL;

    char *job_script = NULL;

    char *base_script =
    "#!/bin/bash\n"
    "%s\n"
    "%s\n"
    "%s\n"
    "%s\n"
    "%s\n"
    "%s\n"
    "%s\n"
    "%s\n"
    "sleep infinity\n";

    FILE *fp = NULL;
    
    prte_err = prte_ras_slurm_make_job_field(fields, str_fields[STR_ACCOUNT], account_format, &account_field, false);

    if(PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    prte_err = prte_ras_slurm_make_job_field(fields, str_fields[STR_PARTITION], partition_format, &partition_field, false);

    if(PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    prte_err = prte_ras_slurm_make_job_field(fields, str_fields[STR_QOS], qos_format, &qos_field, false);

    if(PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    prte_err = prte_ras_slurm_make_job_field(fields, str_fields[STR_CWD], cwd_format, &cwd_field, false);

    if(PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    prte_err = prte_ras_slurm_make_job_field(fields, prte_request_fields[PRTE_REQUEST_NODES], nodes_format, &nodes_field, false);

    if(PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    /* todo: exclude based on MCA parameters */
    prte_err = prte_ras_slurm_make_job_field(fields, num_obj_fields[NUM_OBJ_MEMORY_PER_CPU],
                                            mem_per_cpu_format, &mem_per_cpu_field, true);

    if (PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    if(0 == strlen(mem_per_cpu_field)) {
        prte_err = prte_ras_slurm_make_job_field(fields, num_obj_fields[NUM_OBJ_MEMORY_PER_NODE],
                                                mem_per_node_format, &mem_per_node_field, true);

        if (PRTE_SUCCESS != prte_err) {
            goto cleanup;
        }
    }
    else {
        mem_per_node_field = strdup("");
        
        if(NULL == mem_per_node_field) {
            prte_err = PRTE_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }
    }

    prte_err = prte_ras_slurm_make_job_field(fields, num_obj_fields[NUM_OBJ_TIME_LIMIT],
                                        time_format, &time_limit_field, true);

    if (PRTE_SUCCESS != prte_err) {
        goto cleanup;
    }

    int rc = asprintf(&job_script, base_script,
    account_field,
    partition_field,
    qos_field,
    cwd_field,
    mem_per_cpu_field,
    mem_per_node_field,
    time_limit_field
    );

    if(0 > rc) {
        prte_err = PRTE_ERR_OUT_OF_RESOURCE;
        goto cleanup;
    }

    fp = popen("sbatch", "w");
    if (NULL == fp) {
        prte_err = PRTE_ERR_FILE_OPEN_FAILURE;
        goto cleanup;
    }

    fputs(job_script, fp);

    rc = pclose(fp);
    fp = NULL;

    if (WIFEXITED(rc)) {
        int exit_code = WEXITSTATUS(rc);

        if(0 != exit_code) {
            PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                        "%s ras:slurm:launch_expander_job: job submission failed!",
                        PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
            prte_err = PRTE_ERR_NOT_AVAILABLE;
            goto cleanup;
        }
    }
    else {
        PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
            "%s ras:slurm:launch_expander_job: job submission exited abnormally",
            PRTE_NAME_PRINT(PRTE_PROC_MY_NAME)));
        prte_err = PRTE_ERR_NOT_AVAILABLE;
        goto cleanup;
    }

    cleanup:

    if(NULL != fp) {
        pclose(fp);
    }

    free(job_script);
    
    free(partition_field);
    free(qos_field);
    free(cwd_field);
    free(mem_per_cpu_field);
    free(mem_per_node_field);
    free(time_limit_field);
    free(account_field);

    if(PRTE_SUCCESS != prte_err) {
        PRTE_ERROR_LOG(prte_err);
    }

    return prte_err;
}