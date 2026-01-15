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
 * Copyright (c) 2025      Barcelona Supercomputing Center (BSC-CNS).
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
static int prte_ras_slurm_find_next_quoted(const char *str_in, const char **start_entry, const char **end_entry);

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
    char *slurm_jobid;

    if (NULL == (slurm_jobid = getenv("SLURM_JOBID"))) {
        req->pstatus = PMIX_ERROR;
        return;
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
                *start_entry = str_in;
                start_quote_found = true;
                backslash_count = 0;
            }
            else
            {
                *end_entry = *str_in;
                (*str_in)++;
                return PRTE_SUCCESS;
            }
        } else if ('\\' == *str_in) {
            backslash_count++;
        } else {
            backslash_count = 0;
        }
        (*str_in)++;
    }

    return PRTE_ERR_NOT_FOUND;
}

/*
 * Parse a null-terminated or line-break terminated input text
 * and find the first entry deliminated by a pair of unescaped 
 * quotes inside it. Advance the input string address one step
 * after the end quote.
 * 
 * @param **str_in       Address of a `const char *` pointing to the input text, which will be advanced.
 * @param **start_quote  Address of a `const char *` that will receive the start quote address.
 * @param **end_quote    Address of a `const char *` that will receive the end quote address.
 */
static int prte_ras_slurm_find_next_numeric(const char **str_in, const char **numeric_out)
{
    if(!str_in || !*str_in || !start_quote || !end_quote) {
        PRTE_ERROR_LOG(PRTE_ERR_BAD_PARAM);
        return PRTE_ERR_BAD_PARAM;
    }

    return PRTE_ERR_NOT_FOUND;
}


static int prte_ras_slurm_find_fields(pmix_list_t *fields)
{
    int err = PRTE_SUCCESS;

    char *slurm_jobid;
    if (NULL == (slurm_jobid = getenv("SLURM_JOBID"))) {
        PRTE_ERROR_LOG(PRTE_ERR_NOT_FOUND);
        return PRTE_ERR_NOT_FOUND;
    }

    char const * const jobs_field = "jobs";
    bool job_section_found = false;

    const char *const str_fields[] = {
        "account",
        "partition",
        "qos",
        "current_working_directory",
    };

    const char *const num_fields[] {
        "end_time",
    }

    const char *const var_fields[] {
        "memory_per_cpu",
        "memory_per_node",
    }

    struct var_structure {
        bool set;
        bool infinite;
        char* number;
    }

    const char *const var_subfields[] {
        "set",
        "infinite",
        "number",
    }

    char const * const str_marker = "str"; 
    char const * const num_marker = "num"; 
    char const * const var_marker = "var";

    pmix_kval_t *kv;
    pmix_hash_table_t table;

    PMIX_CONSTRUCT(&table, pmix_hash_table_t);

    size_t str_fields_len = sizeof(str_fields) / sizeof(str_fields[0]);
    size_t num_fields_len = sizeof(num_fields) / sizeof(num_fields[0]);
    size_t var_fields_len = sizeof(var_fields) / sizeof(var_fields[0]);
    size_t total_fields_len = str_fields_len + num_fields_len + var_fields_len;

    pmix_hash_table_init(&table, total_fields_len);

    /* todo: exclude fields based on MCA params */

    for(size_t i = 0; i < str_fields_len; i++) {
        char *key = strdup(str_fields[i]);

        if(NULL == key) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }

        /* set value to the specific type pointer, so we know when a field is unset */
        pmix_hash_table_set_value_ptr(&table, key,
                              strlen(key), str_marker);
    }

    for(size_t i = 0; i < num_fields_len; i++) {
        char *key = strdup(num_fields[i]);

        if(NULL == key) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }

        pmix_hash_table_set_value_ptr(&table, key,
                                strlen(key), num_marker);
    }

    for(size_t i = 0; i < var_fields_len; i++) {
        char *key = strdup(var_fields[i]);

        if(NULL == key) {
            err = PRTE_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }

        pmix_hash_table_set_value_ptr(&table, key,
                                strlen(key), var_marker);
    }

    FILE *fp;

    /* TODO: how about Slurm-side errors */

    fp = popen("cat example_slurm_json.txt", "r");

    if(!fp) {
        printf("popen error\n");
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    int count_found = 0;

    bool jobs_section_found = false;

    bool looking_for_key = true;
    bool looking_for_colon = false;

    const char *open_quote = NULL;
    const char *close_quote = NULL;

    const char *key_ptr = NULL;
    const char *val_ptr = NULL;

    /* output probably consists of multiple lines, but 
     * would still be valid JSON even if one big line */
    while (-1 != (read = getline(&line, &len, fp))) {

        /* remove any whitespace from read line, as JSON does too */

        char *line_read = line;
        char *line_write = line;

        while('\0' != *line_read)
        {
            if(!isspace((unsigned char)*line_read)) {
                *line_write++ = *line_read;
            }

            line_read++;
        }

        *line_write = '\0';

        /* process non-whitespace line */

        while('\0' != *line) {

            /* look for keys in unescaped quotes */
            if(looking_for_key && !looking_for_colon
                && PRTE_ERR_NOT_FOUND != prte_ras_slurm_find_next_quoted(&line, &open_quote, &close_quote)) {
            
                    const char * start_text = open_quote+1;
                    size_t len = close_quote - start_text;

                    key_ptr = strndup(start_text, len);

                    if(NULL == key_ptr) {
                        err = PRTE_ERR_OUT_OF_RESOURCE;
                        goto cleanup;
                    }

                    looking_for_colon = true;
            }

            else if(looking_for_colon) {

                /* we know *line is not the nullchar here */

                looking_for_colon = false;

                /* either we found a colon or this is not a key */
                if(*line != ":")
                {
                    /* keep searching for keys */
                    continue;
                }

                looking_for_key = false;
            }

            /* we have a key; check if it is of interest */
            else if(!looking_for_key) {

                size_t key_len = strlen(key_ptr);

                /* don't bother checking for anything else until we have found the jobs section */
                if(!jobs_section_found && 0 == strcmp(jobs_field, key_ptr)) {
                    key_ptr = NULL;
                    free(key_ptr);
                    jobs_section_found = true;
                    looking_for_key = true;
                    continue;
                }

                else if (PMIX_SUCCESS == pmix_hash_table_get_value_ptr(&table, key_ptr,
                                        len, (void**)&val)) {
                    /* string key */
                    if(val == str_marker) {

                        /* failed to find a corresponding value for the string key */
                        if(*line != '"'
                            || PRTE_ERR_NOT_FOUND == prte_ras_slurm_find_next_quoted(line, &open_quote, &close_quote)) {
                            err = PRTE_ERR_NOT_FOUND;
                            goto cleanup; 
                        }

                        const char * start_text = open_quote+1;
                        len = close_quote-start_text;

                        val_ptr = strndup(start_text, len);

                        if(NULL == val_ptr) {
                            err = PRTE_ERR_OUT_OF_RESOURCE;
                            goto cleanup;
                        }

                        pmix_hash_table_set_value_ptr(&table, key_ptr,
                            strlen(key_ptr), val_ptr);

                        free(key_ptr);
                        free(val_ptr);
                        key_ptr = NULL;
                        val_ptr = NULL;

                        /* extracted the string */
                        count_found++;
                        looking_for_key = true;
                    }
                    
                    else if(val == num_marker)
                    {
                        char *line_start = line;
                        int len = 0;

                        /* numeric key */
                        while (isdigit((unsigned char)*line)) {
                            len++;
                            line++;
                        }

                        

                    }
                }
            }

            
    }

        
    }

    cleanup:

    if(PRTE_SUCCESS != err) {
        PRTE_ERROR_LOG(err);
    }

    if(NULL != key_ptr) {
        free(key_ptr);
    }

    if(NULL != val_ptr) {
        free(val_ptr);
    }

    pmix_hash_table_iterator_t iter;
    void *key;
    size_t keylen;
    void *value;

    PMIX_HASH_TABLE_FOREACH(&iter, &table, key, keylen, value) {
        free(key);

        if(value != str_marker && value != num_marker && value != var_marker) {
            free(value);
        }
    }

    PMIX_DESTRUCT(&table);
    return err;

}