/*
 * Copyright (c) 2026      Barcelona Supercomputing Center (BSC-CNS).
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* Each allocation the scheduler grants is a session marked with our component
 * name. As in ras/slurm, its nodes stay in the general pool (node->session is
 * NULL); membership is kept in session->nodes and PRTE_NODE_ALLOC_ID. */

#include "prte_config.h"
#include "constants.h"
#include "types.h"

#include <stdlib.h>
#include <string.h>

#include "src/class/pmix_list.h"
#include "src/mca/plm/plm_types.h"
#include "src/mca/state/state.h"
#include "src/prted/pmix/pmix_server_internal.h"
#include "src/runtime/prte_globals.h"
#include "src/util/attr.h"
#include "src/util/dash_host/dash_host.h"
#include "src/util/name_fns.h"
#include "src/util/proc_info.h"

#include "src/mca/ras/base/base.h"
#include "ras_pmix.h"

/* nodes of one session being released */
typedef struct {
    pmix_list_item_t super;
    prte_session_t *session;    /* retained */
    char **nodes;
} ras_pmix_return_t;

static void rtcon(ras_pmix_return_t *p)
{
    p->session = NULL;
    p->nodes = NULL;
}
static void rtdes(ras_pmix_return_t *p)
{
    if (NULL != p->session) {
        PMIX_RELEASE(p->session);
    }
    PMIx_Argv_free(p->nodes);
}
static PMIX_CLASS_INSTANCE(ras_pmix_return_t, pmix_list_item_t, rtcon, rtdes);

/* a release waiting for its shrink campaign; the scheduler is told when it
 * completes */
typedef struct {
    pmix_list_item_t super;
    prte_shrink_campaign_t *campaign;
    pmix_list_t returns;
} ras_pmix_release_t;

static void rlcon(ras_pmix_release_t *p)
{
    p->campaign = NULL;
    PMIX_CONSTRUCT(&p->returns, pmix_list_t);
}
static void rldes(ras_pmix_release_t *p)
{
    PMIX_LIST_DESTRUCT(&p->returns);
}
static PMIX_CLASS_INSTANCE(ras_pmix_release_t, pmix_list_item_t, rlcon, rldes);

static pmix_list_t releases;
static bool releases_constructed = false;

static uint32_t session_counter = 0;

void prte_ras_pmix_alloc_init(void)
{
    PMIX_CONSTRUCT(&releases, pmix_list_t);
    releases_constructed = true;
}

void prte_ras_pmix_alloc_finalize(void)
{
    if (releases_constructed) {
        PMIX_LIST_DESTRUCT(&releases);
        releases_constructed = false;
    }
}

static bool is_ours(prte_session_t *session)
{
    return NULL != session && NULL != session->alloc_module &&
           0 == strcmp(session->alloc_module,
                       prte_mca_ras_pmix_component.super.pmix_mca_component_name);
}

static bool is_head_node(prte_node_t *node)
{
    return prte_quickmatch(node, prte_process_info.nodename);
}

static bool argv_has(char **argv, const char *s)
{
    for (int i = 0; NULL != argv && NULL != argv[i]; i++) {
        if (0 == strcmp(argv[i], s)) {
            return true;
        }
    }
    return false;
}

static bool node_is_releasing(const char *name)
{
    ras_pmix_release_t *rel;
    ras_pmix_return_t *ret;

    PMIX_LIST_FOREACH(rel, &releases, ras_pmix_release_t) {
        PMIX_LIST_FOREACH(ret, &rel->returns, ras_pmix_return_t) {
            if (argv_has(ret->nodes, name)) {
                return true;
            }
        }
    }
    return false;
}

static prte_session_t *session_of_node(prte_node_t *node)
{
    prte_session_t *session;

    for (int i = 0; i < prte_sessions->size; i++) {
        session = (prte_session_t *) pmix_pointer_array_get_item(prte_sessions, i);
        if (!is_ours(session)) {
            continue;
        }
        for (int k = 0; k < session->nodes->size; k++) {
            if (node == pmix_pointer_array_get_item(session->nodes, k)) {
                return session;
            }
        }
    }
    return NULL;
}

/* alloc_id points into info; the caller frees ndstring */
static pmix_status_t read_grant(pmix_info_t *info, size_t ninfo,
                                const char **alloc_id, char **ndstring)
{
    pmix_status_t rc;

    *alloc_id = NULL;
    *ndstring = NULL;
    for (size_t n = 0; n < ninfo; n++) {
        if (PMIx_Check_key(info[n].key, PMIX_ALLOC_ID)) {
            if (PMIX_STRING != info[n].value.type || NULL == info[n].value.data.string) {
                return PMIX_ERR_BAD_PARAM;
            }
            *alloc_id = info[n].value.data.string;
        } else if (PMIx_Check_key(info[n].key, PMIX_ALLOC_NODE_LIST) && NULL == *ndstring) {
            rc = prte_ras_base_parse_node_list(&info[n], ndstring);
            if (PMIX_SUCCESS != rc) {
                return rc;
            }
        }
    }
    if (NULL == *alloc_id || NULL == *ndstring) {
        free(*ndstring);
        *ndstring = NULL;
        return PMIX_ERR_BAD_PARAM;
    }
    return PMIX_SUCCESS;
}

/* the nodes must already be in the pool */
static int track_allocation(const char *alloc_id, const char *req_id, char **names,
                            bool dynamic, const pmix_proc_t *requestor)
{
    prte_session_t *session;
    prte_node_t *node;
    int rc;

    session = PMIX_NEW(prte_session_t);
    if (NULL == session) {
        return PRTE_ERR_OUT_OF_RESOURCE;
    }
    /* use the scheduler's id when it is a free number, as ras/slurm does with
     * job ids: plm/slurm passes the session id as --jobid */
    session->session_id = UINT32_MAX;
    {
        char *end = NULL;
        unsigned long long v = strtoull(alloc_id, &end, 10);

        if ('\0' != alloc_id[0] && '\0' == *end && v < UINT32_MAX &&
            '-' != alloc_id[0] && NULL == prte_get_session_object((uint32_t) v)) {
            session->session_id = (uint32_t) v;
        }
    }
    while (UINT32_MAX == session->session_id ||
           NULL != prte_get_session_object(session->session_id)) {
        session->session_id = ++session_counter;
    }
    session->alloc_refid = strdup(alloc_id);
    session->alloc_module = strdup(prte_mca_ras_pmix_component.super.pmix_mca_component_name);
    if (NULL != req_id) {
        session->user_refid = strdup(req_id);
    }
    if (NULL == session->alloc_refid || NULL == session->alloc_module ||
        (NULL != req_id && NULL == session->user_refid)) {
        PMIX_RELEASE(session);
        return PRTE_ERR_OUT_OF_RESOURCE;
    }
    if (dynamic) {
        PRTE_FLAG_SET(session, PRTE_SESSION_FLAG_DYNAMIC);
    }
    /* the grow campaign sends PMIX_DVM_IS_READY to this requester */
    if (NULL != requestor) {
        PMIX_XFER_PROCID(&session->requestor, requestor);
    }

    for (int i = 0; NULL != names && NULL != names[i]; i++) {
        node = prte_node_match(NULL, names[i]);
        if (NULL == node) {
            PMIX_RELEASE(session);
            return PRTE_ERR_NOT_FOUND;
        }
        rc = prte_set_attribute(&node->attributes, PRTE_NODE_ALLOC_ID, PRTE_ATTR_LOCAL,
                                &session->session_id, PMIX_UINT32);
        if (PRTE_SUCCESS != rc) {
            PMIX_RELEASE(session);
            return rc;
        }
        PMIX_RETAIN(node);
        if (0 > pmix_pointer_array_add(session->nodes, node)) {
            PMIX_RELEASE(node);
            PMIX_RELEASE(session);
            return PRTE_ERR_OUT_OF_RESOURCE;
        }
    }

    rc = prte_set_session_object(session);
    if (PRTE_SUCCESS != rc) {
        PMIX_RELEASE(session);
        return rc;
    }
    prte_num_allocated_nodes += PMIx_Argv_count(names);
    return PRTE_SUCCESS;
}

static int parse_nodes(char *ndstring, pmix_list_t *nodes, char ***names)
{
    prte_node_t *node;
    int rc;

    rc = prte_util_add_dash_host_nodes(nodes, ndstring);
    if (PRTE_SUCCESS != rc) {
        return rc;
    }
    if (pmix_list_is_empty(nodes)) {
        return PRTE_ERR_NOT_FOUND;
    }
    PMIX_LIST_FOREACH(node, nodes, prte_node_t) {
        PMIx_Argv_append_nosize(names, node->name);
    }
    return PRTE_SUCCESS;
}

int prte_ras_pmix_adopt_allocation(prte_job_t *jdata, pmix_info_t *info, size_t ninfo)
{
    const char *alloc_id;
    char *ndstring = NULL;
    char **names = NULL;
    char *id = NULL;
    pmix_list_t nodes;
    pmix_status_t prc;
    int rc;

    prc = read_grant(info, ninfo, &alloc_id, &ndstring);
    if (PMIX_SUCCESS != prc) {
        pmix_output(0, "ras:pmix: the scheduler's answer to the DVM's allocation request "
                       "names no PMIX_ALLOC_ID or no PMIX_ALLOC_NODE_LIST");
        return prte_pmix_convert_status(prc);
    }
    id = strdup(alloc_id);
    if (NULL == id) {
        free(ndstring);
        return PRTE_ERR_OUT_OF_RESOURCE;
    }

    PMIX_CONSTRUCT(&nodes, pmix_list_t);
    rc = parse_nodes(ndstring, &nodes, &names);
    free(ndstring);
    if (PRTE_SUCCESS != rc) {
        PMIX_LIST_DESTRUCT(&nodes);
        PMIx_Argv_free(names);
        free(id);
        return rc;
    }
    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                         "%s ras:pmix: the scheduler granted allocation %s with %d node(s)",
                         PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), id, PMIx_Argv_count(names)));

    /* the state change it posts runs after the session below exists */
    prte_ras_base_allocation_granted(jdata, &nodes);
    rc = track_allocation(id, NULL, names, false, NULL);
    if (PRTE_SUCCESS != rc) {
        PRTE_ERROR_LOG(rc);
        PRTE_ACTIVATE_JOB_STATE(jdata, PRTE_JOB_STATE_ALLOC_FAILED);
    }
    PMIx_Argv_free(names);
    free(id);
    return PRTE_SUCCESS;
}

bool prte_ras_pmix_names_reservation(prte_pmix_server_req_t *req)
{
    prte_session_t *session = NULL;

    for (size_t n = 0; n < req->ninfo && NULL == session; n++) {
        if (PMIX_STRING != req->info[n].value.type) {
            continue;
        }
        if (PMIx_Check_key(req->info[n].key, PMIX_ALLOC_ID)) {
            session = prte_get_session_object_from_id(req->info[n].value.data.string);
        } else if (PMIx_Check_key(req->info[n].key, PMIX_ALLOC_REQ_ID)) {
            session = prte_get_session_object_from_refid(req->info[n].value.data.string);
        }
    }
    return NULL != session && !is_ours(session);
}

void prte_ras_pmix_grow(prte_pmix_server_req_t *req)
{
    const char *granted;
    char *alloc_id = NULL, *req_id = NULL, *ndstring = NULL;
    char **names = NULL;
    pmix_info_t *rinfo;
    prte_node_t *node, *next, *existing;
    pmix_list_t nodes;
    pmix_status_t prc;
    int rc;

    prc = read_grant(req->info, req->ninfo, &granted, &ndstring);
    if (PMIX_SUCCESS != prc) {
        pmix_output(0, "ras:pmix: the scheduler's answer to a grow names no "
                       "PMIX_ALLOC_ID or no PMIX_ALLOC_NODE_LIST");
        req->pstatus = prc;
        return;
    }
    alloc_id = strdup(granted);
    for (size_t n = 0; n < req->ninfo; n++) {
        if (PMIx_Check_key(req->info[n].key, PMIX_ALLOC_REQ_ID) &&
            PMIX_STRING == req->info[n].value.type) {
            req_id = strdup(req->info[n].value.data.string);
            break;
        }
    }
    PMIX_CONSTRUCT(&nodes, pmix_list_t);
    if (NULL == alloc_id) {
        rc = PRTE_ERR_OUT_OF_RESOURCE;
        goto giveback;
    }

    /* every grow is a new allocation */
    if (NULL != prte_get_session_object_from_id(alloc_id)) {
        pmix_output(0, "ras:pmix: the scheduler granted a grow as allocation %s, "
                       "which this DVM already holds", alloc_id);
        rc = PRTE_EXISTS;
        goto giveback;
    }

    rc = parse_nodes(ndstring, &nodes, &names);
    if (PRTE_SUCCESS != rc) {
        goto giveback;
    }

    /* a node granted again after a release is still in the pool: reuse it */
    PMIX_LIST_FOREACH_SAFE(node, next, &nodes, prte_node_t) {
        existing = prte_node_match(NULL, node->name);
        if (NULL == existing) {
            node->state = PRTE_NODE_STATE_ADDED;
            continue;
        }
        if (NULL != existing->daemon ||
            PRTE_FLAG_TEST(existing, PRTE_NODE_FLAG_DAEMON_LAUNCHED)) {
            pmix_output(0, "ras:pmix: the scheduler granted node %s, which is "
                           "already part of the DVM", node->name);
            rc = PRTE_EXISTS;
            goto giveback;
        }
        if (PRTE_FLAG_TEST(node, PRTE_NODE_FLAG_SLOTS_GIVEN)) {
            existing->slots = node->slots;
            PRTE_FLAG_SET(existing, PRTE_NODE_FLAG_SLOTS_GIVEN);
        }
        existing->slots_inuse = 0;
        existing->state = PRTE_NODE_STATE_ADDED;
        pmix_list_remove_item(&nodes, &node->super);
        PMIX_RELEASE(node);
    }

    rc = prte_ras_base_node_insert(&nodes, NULL);
    if (PRTE_SUCCESS != rc) {
        goto giveback;
    }
    rc = track_allocation(alloc_id, req_id, names, true, &req->tproc);
    if (PRTE_SUCCESS != rc) {
        goto giveback;
    }
    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                         "%s ras:pmix: grow granted as allocation %s with %d node(s)",
                         PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), alloc_id,
                         PMIx_Argv_count(names)));

    PMIX_INFO_CREATE(rinfo, 2);
    if (NULL != rinfo) {
        PMIX_INFO_LOAD(&rinfo[0], PMIX_ALLOC_ID, alloc_id, PMIX_STRING);
        PMIX_INFO_LOAD(&rinfo[1], PMIX_RM_NAME,
                       prte_mca_ras_pmix_component.super.pmix_mca_component_name, PMIX_STRING);
        if (req->copy && NULL != req->info) {
            PMIX_INFO_FREE(req->info, req->ninfo);
        }
        req->info = rinfo;
        req->ninfo = 2;
        req->copy = true;
    }

    prte_ras_base_activate_dvm_grow();
    req->pstatus = PMIX_SUCCESS;
#if PRTE_HAVE_DVM_MOD_EVENTS
    /* the grow campaign's PMIX_DVM_IS_READY completes it */
    if (prte_elastic_mode) {
        req->pstatus = PMIX_OPERATION_IN_PROGRESS;
    }
#endif
    PMIX_LIST_DESTRUCT(&nodes);
    PMIx_Argv_free(names);
    free(ndstring);
    free(alloc_id);
    free(req_id);
    return;

giveback:
    /* we cannot use what was granted, so return it */
    req->pstatus = prte_pmix_convert_rc(rc);
    if (NULL != alloc_id && PRTE_EXISTS != rc) {
        char **all = NULL;
        if (NULL != ndstring) {
            all = PMIx_Argv_split(ndstring, ',');
        }
        prte_ras_pmix_give_back(alloc_id, all);
        PMIx_Argv_free(all);
    }
    PMIX_LIST_DESTRUCT(&nodes);
    PMIx_Argv_free(names);
    free(ndstring);
    free(alloc_id);
    free(req_id);
}

typedef struct {
    char *alloc_id;
    char **nodes;
    uint64_t count;
    bool have_count;
} release_spec_t;

static pmix_status_t read_release(prte_pmix_server_req_t *req, release_spec_t *spec)
{
    char *ndstring;
    pmix_status_t rc;

    memset(spec, 0, sizeof(*spec));
    for (size_t n = 0; n < req->ninfo; n++) {
        if (PMIx_Check_key(req->info[n].key, PMIX_ALLOC_ID)) {
            if (PMIX_STRING != req->info[n].value.type) {
                return PMIX_ERR_BAD_PARAM;
            }
            spec->alloc_id = req->info[n].value.data.string;
        } else if (PMIx_Check_key(req->info[n].key, PMIX_ALLOC_NODE_LIST)) {
            rc = prte_ras_base_parse_node_list(&req->info[n], &ndstring);
            if (PMIX_SUCCESS != rc) {
                return rc;
            }
            PMIx_Argv_free(spec->nodes);
            spec->nodes = PMIx_Argv_split(ndstring, ',');
            free(ndstring);
        } else if (PMIx_Check_key(req->info[n].key, PMIX_ALLOC_NUM_NODES)) {
            if (PMIX_SUCCESS != PMIx_Value_get_number(&req->info[n].value, &spec->count,
                                                      PMIX_UINT64)) {
                return PMIX_ERR_BAD_PARAM;
            }
            spec->have_count = true;
        }
    }
    return PMIX_SUCCESS;
}

bool prte_ras_pmix_release_is_ours(prte_pmix_server_req_t *req)
{
    release_spec_t spec;
    prte_node_t *node;
    bool ours;

    if (PMIX_SUCCESS != read_release(req, &spec)) {
        PMIx_Argv_free(spec.nodes);
        return false;
    }
    if (NULL != spec.alloc_id) {
        ours = is_ours(prte_get_session_object_from_id(spec.alloc_id));
    } else if (NULL != spec.nodes) {
        ours = true;
        for (int i = 0; ours && NULL != spec.nodes[i]; i++) {
            node = prte_node_match(NULL, spec.nodes[i]);
            ours = (NULL != node && NULL != session_of_node(node));
        }
    } else {
        ours = spec.have_count;
    }
    PMIx_Argv_free(spec.nodes);
    return ours;
}

static ras_pmix_return_t *return_for(ras_pmix_release_t *rel, prte_session_t *session)
{
    ras_pmix_return_t *ret;

    PMIX_LIST_FOREACH(ret, &rel->returns, ras_pmix_return_t) {
        if (ret->session == session) {
            return ret;
        }
    }
    ret = PMIX_NEW(ras_pmix_return_t);
    if (NULL == ret) {
        return NULL;
    }
    PMIX_RETAIN(session);
    ret->session = session;
    pmix_list_append(&rel->returns, &ret->super);
    return ret;
}

static int plan_node(ras_pmix_release_t *rel, prte_node_t *node)
{
    prte_session_t *session = session_of_node(node);
    ras_pmix_return_t *ret;

    if (NULL == session) {
        return PRTE_ERR_NOT_FOUND;
    }
    if (is_head_node(node)) {
        return PRTE_ERR_BAD_PARAM;
    }
    if (NULL == node->daemon) {
        return PRTE_ERR_NOT_FOUND;
    }
    if (node_is_releasing(node->name)) {
        return PRTE_ERR_RESOURCE_BUSY;
    }
    ret = return_for(rel, session);
    if (NULL == ret) {
        return PRTE_ERR_OUT_OF_RESOURCE;
    }
    if (!argv_has(ret->nodes, node->name)) {
        PMIx_Argv_append_nosize(&ret->nodes, node->name);
    }
    return PRTE_SUCCESS;
}

/* take nodes from the most recent allocations first, as ras/slurm does */
static int plan_count(ras_pmix_release_t *rel, uint64_t count)
{
    prte_session_t *session, *newest;
    prte_node_t *node;
    uint32_t below = UINT32_MAX;
    uint64_t taken = 0;
    int rc;

    while (taken < count) {
        newest = NULL;
        for (int i = 0; i < prte_sessions->size; i++) {
            session = (prte_session_t *) pmix_pointer_array_get_item(prte_sessions, i);
            if (is_ours(session) && session->acquisition < below &&
                (NULL == newest || session->acquisition > newest->acquisition)) {
                newest = session;
            }
        }
        if (NULL == newest) {
            return PRTE_ERR_OUT_OF_RESOURCE;
        }
        below = newest->acquisition;
        for (int k = newest->nodes->size - 1; 0 <= k && taken < count; k--) {
            node = (prte_node_t *) pmix_pointer_array_get_item(newest->nodes, k);
            if (NULL == node || is_head_node(node) || NULL == node->daemon ||
                node_is_releasing(node->name)) {
                continue;
            }
            rc = plan_node(rel, node);
            if (PRTE_SUCCESS != rc) {
                return rc;
            }
            taken++;
        }
    }
    return PRTE_SUCCESS;
}

static int plan_release(ras_pmix_release_t *rel, release_spec_t *spec)
{
    prte_session_t *session;
    prte_node_t *node;
    int rc;

    if (NULL != spec->alloc_id) {
        session = prte_get_session_object_from_id(spec->alloc_id);
        if (NULL == session) {
            return PRTE_ERR_NOT_FOUND;
        }
        /* the initial allocation holds the HNP */
        if (!PRTE_FLAG_TEST(session, PRTE_SESSION_FLAG_DYNAMIC)) {
            return PRTE_ERR_NOT_SUPPORTED;
        }
        for (int k = 0; k < session->nodes->size; k++) {
            node = (prte_node_t *) pmix_pointer_array_get_item(session->nodes, k);
            if (NULL == node) {
                continue;
            }
            rc = plan_node(rel, node);
            if (PRTE_SUCCESS != rc) {
                return rc;
            }
        }
        return PRTE_SUCCESS;
    }
    if (NULL != spec->nodes) {
        for (int i = 0; NULL != spec->nodes[i]; i++) {
            node = prte_node_match(NULL, spec->nodes[i]);
            if (NULL == node) {
                return PRTE_ERR_NOT_FOUND;
            }
            rc = plan_node(rel, node);
            if (PRTE_SUCCESS != rc) {
                return rc;
            }
        }
        return PRTE_SUCCESS;
    }
    return plan_count(rel, spec->count);
}

pmix_status_t prte_ras_pmix_serve_release(prte_pmix_server_req_t *req)
{
    release_spec_t spec;
    ras_pmix_release_t *rel;
    ras_pmix_return_t *ret;
    prte_shrink_campaign_t *campaign = NULL;
    pmix_rank_t *ranks = NULL;
    prte_node_t *node;
    pmix_status_t prc;
    int32_t nranks = 0;
    int rc;

    prc = read_release(req, &spec);
    if (PMIX_SUCCESS != prc) {
        PMIx_Argv_free(spec.nodes);
        return prc;
    }
    rel = PMIX_NEW(ras_pmix_release_t);
    if (NULL == rel) {
        PMIx_Argv_free(spec.nodes);
        return PMIX_ERR_NOMEM;
    }
    rc = plan_release(rel, &spec);
    PMIx_Argv_free(spec.nodes);
    if (PRTE_SUCCESS == rc && pmix_list_is_empty(&rel->returns)) {
        rc = PRTE_ERR_NOT_FOUND;
    }
    if (PRTE_SUCCESS != rc) {
        PMIX_RELEASE(rel);
        return prte_pmix_convert_rc(rc);
    }

    PMIX_LIST_FOREACH(ret, &rel->returns, ras_pmix_return_t) {
        nranks += PMIx_Argv_count(ret->nodes);
    }
    ranks = (pmix_rank_t *) malloc(nranks * sizeof(pmix_rank_t));
    if (NULL == ranks) {
        PMIX_RELEASE(rel);
        return PMIX_ERR_NOMEM;
    }
    nranks = 0;
    PMIX_LIST_FOREACH(ret, &rel->returns, ras_pmix_return_t) {
        for (int i = 0; NULL != ret->nodes[i]; i++) {
            node = prte_node_match(NULL, ret->nodes[i]);
            ranks[nranks++] = node->daemon->name.rank;
        }
    }

    rc = prte_ras_base_prepare_dvm_shrink(req, ranks, nranks, &campaign);
    free(ranks);
    if (PRTE_SUCCESS != rc || NULL == campaign) {
        PMIX_RELEASE(rel);
        return prte_pmix_convert_rc((PRTE_SUCCESS == rc) ? PRTE_ERR_BAD_PARAM : rc);
    }
    rel->campaign = campaign;
    pmix_list_append(&releases, &rel->super);
    rc = prte_ras_base_commit_dvm_shrink(campaign);
    if (PRTE_SUCCESS != rc) {
        /* the commit aborted the campaign */
        pmix_list_remove_item(&releases, &rel->super);
        PMIX_RELEASE(rel);
        return prte_pmix_convert_rc(rc);
    }
    PMIX_OUTPUT_VERBOSE((1, prte_ras_base_framework.framework_output,
                         "%s ras:pmix: releasing %d node(s), handed back when the shrink drains",
                         PRTE_NAME_PRINT(PRTE_PROC_MY_NAME), (int) nranks));

#if PRTE_HAVE_DVM_MOD_EVENTS
    req->pstatus = PMIX_OPERATION_IN_PROGRESS;
#else
    req->pstatus = PMIX_SUCCESS;
#endif
    if (NULL != req->infocbfunc) {
        req->infocbfunc(req->pstatus, req->info, req->ninfo, req->cbdata,
                        prte_pmix_server_req_release, req);
    } else {
        pmix_pointer_array_set_item(&prte_pmix_server_globals.local_reqs, req->local_index,
                                    NULL);
        PMIX_RELEASE(req);
    }
    return PMIX_SUCCESS;
}

static int detach_nodes(prte_session_t *session, char **names)
{
    prte_node_t *node;
    int dropped = 0;

    for (int k = 0; k < session->nodes->size; k++) {
        node = (prte_node_t *) pmix_pointer_array_get_item(session->nodes, k);
        if (NULL == node || NULL == node->name || !argv_has(names, node->name)) {
            continue;
        }
        pmix_pointer_array_set_item(session->nodes, k, NULL);
        PMIX_RELEASE(node);
        dropped++;
    }
    return dropped;
}

static bool session_is_empty(prte_session_t *session)
{
    for (int k = 0; k < session->nodes->size; k++) {
        if (NULL != pmix_pointer_array_get_item(session->nodes, k)) {
            return false;
        }
    }
    return true;
}

void prte_ras_pmix_shrink_complete(prte_shrink_campaign_t *campaign)
{
    ras_pmix_release_t *rel, *found = NULL;
    ras_pmix_return_t *ret;
    prte_proc_t *dproc;
    prte_job_t *daemons;

    PMIX_LIST_FOREACH(rel, &releases, ras_pmix_release_t) {
        if (rel->campaign == campaign) {
            found = rel;
            break;
        }
    }
    if (NULL == found) {
        return;
    }
    pmix_list_remove_item(&releases, &found->super);

    /* the nodes go back to the scheduler: keep them out of the pool */
    daemons = prte_get_job_data_object(PRTE_PROC_MY_NAME->nspace);
    for (int i = 0; NULL != daemons && i < campaign->ntargets; i++) {
        dproc = (prte_proc_t *) pmix_pointer_array_get_item(daemons->procs,
                                                            campaign->targets[i]);
        if (NULL != dproc && NULL != dproc->node) {
            dproc->node->state = PRTE_NODE_STATE_NOT_INCLUDED;
        }
    }

    PMIX_LIST_FOREACH(ret, &found->returns, ras_pmix_return_t) {
        prte_session_t *session = ret->session;

        prte_num_allocated_nodes -= detach_nodes(session, ret->nodes);
        prte_ras_pmix_give_back(session->alloc_refid, ret->nodes);
        if (PRTE_FLAG_TEST(session, PRTE_SESSION_FLAG_DYNAMIC) &&
            session_is_empty(session)) {
            PMIX_RELEASE(session);
        }
    }
    PMIX_RELEASE(found);
}
