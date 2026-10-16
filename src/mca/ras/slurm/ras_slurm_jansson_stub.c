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

#include "ras_slurm.h"

/**
 * Check if we have the Jansson library available in compilation
 */
bool prte_ras_slurm_have_jansson(void)
{
    return false;
}

/*
 * Extract SLURM job fields; returns PRTE_ERR_NOT_AVAILABLE if built without Jansson.
 */
int prte_ras_slurm_extract_job_fields(FILE *fp, pmix_hash_table_t *values_table,
                                      time_t *start_time, time_t *end_time)
{
    PRTE_HIDE_UNUSED_PARAMS(fp, values_table, start_time, end_time);
    pmix_output(0, "ras:slurm:extract_job_fields: "
                "Jansson support is required but not enabled in this build");
    return PRTE_ERR_NOT_AVAILABLE;
}

/**
 * Add new SLURM job resources; returns PRTE_ERR_NOT_AVAILABLE if built without Jansson.
 */
int prte_ras_slurm_add_modified_resources(FILE *fp, pmix_list_t *node_list)
{
    PRTE_HIDE_UNUSED_PARAMS(fp, node_list);

    pmix_output(0, "ras:slurm:add_modified_resources: "
                "Jansson support is required but not enabled in this build");
    return PRTE_ERR_NOT_AVAILABLE;
}

/**
 * Wait for SLURM job resources; returns PRTE_ERR_NOT_AVAILABLE if built without Jansson.
 */
int prte_ras_slurm_check_resources(FILE *fp, time_t *start_time, time_t *end_time)
{
    PRTE_HIDE_UNUSED_PARAMS(fp, start_time, end_time);
    pmix_output(0, "ras:slurm:wait_resources: "
                "Jansson support is required but not enabled in this build");
    return PRTE_ERR_NOT_AVAILABLE;
}

/**
 * Read a job's start/end times; returns PRTE_ERR_NOT_AVAILABLE if built without Jansson.
 */
int prte_ras_slurm_get_job_times(FILE *fp, time_t *start_time, time_t *end_time)
{
    PRTE_HIDE_UNUSED_PARAMS(fp, start_time, end_time);
    pmix_output(0, "ras:slurm:get_job_times: "
                "Jansson support is required but not enabled in this build");
    return PRTE_ERR_NOT_AVAILABLE;
}
