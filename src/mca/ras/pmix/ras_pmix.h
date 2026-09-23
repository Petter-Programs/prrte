/*
 * Copyright (c) 2011-2020 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2015-2019 Intel, Inc.  All rights reserved.
 * Copyright (c) 2019      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2021-2026 Nanook Consulting  All rights reserved.
 * Copyright (c) 2026      Barcelona Supercomputing Center (BSC-CNS).
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef PRTE_RAS_PMIX_H
#define PRTE_RAS_PMIX_H

#include "prte_config.h"
#include "src/mca/ras/base/base.h"
#include "src/mca/ras/ras.h"

BEGIN_C_DECLS

struct prte_ras_pmix_component_t {
    prte_ras_base_component_t super;
    bool connect_to_system_scheduler;
    pmix_proc_t server;
    char *uri;
    char *connection_order;
    pid_t server_pid;
    char *server_host;
    uint32_t max_retries;
    uint32_t retry_delay;
};
typedef struct prte_ras_pmix_component_t prte_ras_pmix_component_t;

PRTE_EXPORT extern prte_ras_pmix_component_t prte_mca_ras_pmix_component;
PRTE_EXPORT extern prte_ras_base_module_t prte_ras_pmix_module;

int prte_ras_pmix_give_back(const char *alloc_id, char **nodes);

/* ras_pmix_alloc.c */
void prte_ras_pmix_alloc_init(void);
void prte_ras_pmix_alloc_finalize(void);
int prte_ras_pmix_adopt_allocation(prte_job_t *jdata, pmix_info_t *info, size_t ninfo);
bool prte_ras_pmix_names_reservation(prte_pmix_server_req_t *req);
void prte_ras_pmix_grow(prte_pmix_server_req_t *req);
bool prte_ras_pmix_release_is_ours(prte_pmix_server_req_t *req);
pmix_status_t prte_ras_pmix_serve_release(prte_pmix_server_req_t *req);
void prte_ras_pmix_shrink_complete(prte_shrink_campaign_t *campaign);

END_C_DECLS

#endif
