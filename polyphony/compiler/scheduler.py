﻿from collections import defaultdict, deque
from .common import fail, warn
from .errors import Errors, Warnings
from .latency import get_latency
from .irvisitor import IRVisitor
from .ir import *
from .irhelper import has_exclusive_function
from .latency import CALL_MINIMUM_STEP
from .utils import unique
from .scope import Scope
from logging import getLogger
logger = getLogger(__name__)

MAX_FUNC_UNIT = 10


class Scheduler(object):
    def __init__(self):
        self.done_blocks = []

    def schedule(self, scope):
        if scope.is_namespace() or scope.is_class() or scope.is_lib():
            return
        self.scope = scope
        for dfg in self.scope.dfgs(bottom_up=True):
            if dfg.parent and dfg.synth_params['scheduling'] == 'pipeline':
                scheduler_impl = PipelineScheduler()
            else:
                scheduler_impl = BlockBoundedListScheduler()
            scheduler_impl.schedule(scope, dfg)


class SchedulerImpl(object):
    def __init__(self):
        self.res_tables = {}
        self.node_latency_map = {}  # {node:(max, min, actual)}
        self.node_seq_latency_map = {}
        self.all_paths = []
        self.res_extractor = None

    def schedule(self, scope, dfg):
        self.scope = scope
        logger.log(0, '_schedule dfg')
        sources = dfg.find_src()
        for src in sources:
            src.priority = -1

        self.res_extractor = ResourceExtractor()
        for node in sorted(dfg.traverse_nodes(dfg.succs, sources, [])):
            self.res_extractor.current_node = node
            self.res_extractor.visit(node.tag)

        worklist = deque()
        worklist.append((sources, 0))

        while worklist:
            nodes, prio = worklist.popleft()
            for n in nodes:
                succs, nextprio = self._set_priority(n, prio, dfg)
                if succs:
                    succs = unique(succs)
                    worklist.append((succs, nextprio))
        longest_latency = self._schedule(dfg)
        if longest_latency > CALL_MINIMUM_STEP:
            scope.asap_latency = longest_latency
        else:
            scope.asap_latency = CALL_MINIMUM_STEP

    def _set_priority(self, node, prio, dfg):
        if prio > node.priority:
            node.priority = prio
            logger.debug('update priority ... ' + str(node))
            return (dfg.succs_without_back(node), prio + 1)
        return (None, None)

    def _node_sched_default(self, dfg, node):
        preds = dfg.preds_without_back(node)
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, 'DefUse')
            usedef_preds = dfg.preds_typ_without_back(node, 'UseDef')
            seq_preds = dfg.preds_typ_without_back(node, 'Seq')
            sched_times = []
            if seq_preds:
                latest_node = max(seq_preds, key=lambda p: p.end)
                sched_times.append(latest_node.end)
            if defuse_preds:
                latest_node = max(defuse_preds, key=lambda p: p.end)
                sched_times.append(latest_node.end)
            if usedef_preds:
                preds = usedef_preds
                latest_node = max(preds, key=lambda p: p.begin)
                sched_times.append(latest_node.begin)
            if not sched_times:
                latest_node = max(preds, key=lambda p: p.begin)
                sched_times.append(latest_node.begin)
            scheduled_time = max(sched_times)
            if scheduled_time < 0:
                scheduled_time = 0
        else:
            # source node
            scheduled_time = 0
        return scheduled_time

    def _find_latest_alias(self, dfg, node):
        stm = node.tag
        if not stm.is_a([MOVE, PHIBase]):
            return node
        var = node.tag.dst.symbol() if node.tag.is_a(MOVE) else node.tag.var.symbol()
        if not var.is_alias():
            return node
        succs = dfg.succs_typ_without_back(node, 'DefUse')
        if not succs:
            return node
        nodes = [self._find_latest_alias(dfg, s) for s in succs]
        latest_node = max(nodes, key=lambda p: p.end)
        return latest_node

    def _is_resource_full(self, res, scheduled_resources):
        # TODO:
        if isinstance(res, str):
            return len(scheduled_resources) >= MAX_FUNC_UNIT
        elif isinstance(res, Scope):
            return len(scheduled_resources) >= MAX_FUNC_UNIT
        return 0

    def _str_res(self, res):
        if isinstance(res, str):
            return res
        elif isinstance(res, Scope):
            return res.name

    def _get_earliest_res_free_time(self, node, time, latency):
        resources = self.res_extractor.ops[node].keys()
        #TODO operator chaining?
        #logger.debug(node)
        #logger.debug(resources)
        assert len(resources) <= 1
        if resources:
            res = list(resources)[0]
            if res not in self.res_tables:
                table = defaultdict(list)
                self.res_tables[res] = table
            else:
                table = self.res_tables[res]

            scheduled_resources = table[time]
            if node in scheduled_resources:
                #already scheduled
                return time

            while self._is_resource_full(res, scheduled_resources):
                logger.debug("!!! resource {}'s slot '{}' is full !!!".
                             format(self._str_res(res), time))
                time += 1
                scheduled_resources = table[time]

            node.instance_num = len(scheduled_resources)
            #logger.debug("{} is scheduled to {}, instance_num {}".
            #             format(node, time, node.instance_num))

            # fill scheduled_resources table
            n = latency if latency != 0 else 1
            for i in range(n):
                scheduled_resources = table[time + i]
                scheduled_resources.append(node)
        return time

    def _calc_latency(self, dfg):
        is_minimum = dfg.synth_params['cycle'] == 'minimum'
        for node in dfg.get_priority_ordered_nodes():
            def_l, seq_l = get_latency(node.tag)
            if def_l == 0:
                if is_minimum:
                    self.node_latency_map[node] = (0, 0, 0)
                else:
                    if node.tag.is_a([MOVE, PHIBase]):
                        var = node.tag.dst.symbol() if node.tag.is_a(MOVE) else node.tag.var.symbol()
                        if var.is_condition():
                            self.node_latency_map[node] = (0, 0, 0)
                        else:
                            self.node_latency_map[node] = (1, 0, 1)
                    else:
                        self.node_latency_map[node] = (0, 0, 0)
            else:
                self.node_latency_map[node] = (def_l, def_l, def_l)
            self.node_seq_latency_map[node] = seq_l

    def _adjust_latency(self, paths, expected):
        for path in paths:
            path_latencies = []
            for n in path:
                m, _, _ = self.node_latency_map[n]
                path_latencies.append(m)
            path_latency = sum(path_latencies)
            if expected > path_latency:
                # we don't have to adjust latency
                continue
            diff = path_latency - expected
            fixed = set()
            succeeded = True
            # try to reduce latency
            while diff:
                for i, n in enumerate(path):
                    if n in fixed:
                        continue
                    max_l, min_l, _ = self.node_latency_map[n]
                    if min_l < path_latencies[i]:
                        path_latencies[i] -= 1
                        self.node_latency_map[n] = (max_l, min_l, path_latencies[i])
                        diff -= 1
                    else:
                        fixed.add(n)
                if len(fixed) == len(path):
                    # scheduling has failed
                    succeeded = False
                    break
            if not succeeded:
                return False, expected + diff
        return True, expected

    def _try_adjust_latency(self, dfg, expected):
        for path in dfg.trace_all_paths(lambda n: dfg.succs_typ_without_back(n, 'DefUse')):
            self.all_paths.append(path)
        ret, actual = self._adjust_latency(self.all_paths, expected)
        if not ret:
            assert False, 'scheduling has failed. the cycle must be greater equal {}'.format(actual)

    def _max_latency(self, paths):
        max_latency = 0
        for path in paths:
            path_latencies = []
            for n in path:
                m, _, _ = self.node_latency_map[n]
                path_latencies.append(m)
            path_latency = sum(path_latencies)
            if path_latency > max_latency:
                max_latency = path_latency
        return max_latency

    def _remove_alias_if_needed(self, dfg):
        for n in dfg.nodes:
            if n not in self.node_latency_map:
                continue
            _, min_l, actual = self.node_latency_map[n]
            if min_l == 0 and actual > 0:
                for d in n.defs:
                    if d.is_alias():
                        d.del_tag('alias')

    def _group_nodes_by_block(self, dfg):
        block_nodes = defaultdict(list)
        for node in dfg.get_priority_ordered_nodes():
            block_nodes[node.tag.block].append(node)
        return block_nodes

    def _schedule_cycles(self, dfg):
        self._calc_latency(dfg)
        synth_cycle = dfg.synth_params['cycle']
        if synth_cycle == 'any' or synth_cycle == 'minimum':
            pass
        elif synth_cycle.startswith('less:'):
            extected_latency = int(synth_cycle[len('less:'):])
            self._try_adjust_latency(dfg, extected_latency)
        elif synth_cycle.startswith('greater:'):
            assert False, 'Not Implement Yet'
        else:
            assert False


class BlockBoundedListScheduler(SchedulerImpl):
    def _schedule(self, dfg):
        self._schedule_cycles(dfg)
        self._remove_alias_if_needed(dfg)

        block_nodes = self._group_nodes_by_block(dfg)
        longest_latency = 0
        for block, nodes in block_nodes.items():
            #latency = self._list_schedule(dfg, nodes)
            latency = self._list_schedule_with_block_bound(dfg, nodes, block, 0)
            if longest_latency < latency:
                longest_latency = latency
        return longest_latency

    def _list_schedule(self, dfg, nodes):
        next_candidates = set()
        latency = 0
        for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
            scheduled_time = self._node_sched(dfg, n)
            latency = get_latency(n.tag)
            #detect resource conflict
            scheduled_time = self._get_earliest_res_free_time(n, scheduled_time, latency)
            n.begin = scheduled_time
            n.end = n.begin + latency
            #logger.debug('## SCHEDULED ## ' + str(n))
            succs = dfg.succs_without_back(n)
            next_candidates = next_candidates.union(succs)
            latency = n.end
        if next_candidates:
            return self._list_schedule(dfg, next_candidates)
        else:
            return latency

    def _list_schedule_with_block_bound(self, dfg, nodes, block, longest_latency):
        next_candidates = set()
        for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
            if n.tag.block is not block:
                continue
            scheduled_time = self._node_sched_with_block_bound(dfg, n, block)
            _, _, latency = self.node_latency_map[n]
            #detect resource conflict
            scheduled_time = self._get_earliest_res_free_time(n, scheduled_time, latency)
            n.begin = scheduled_time
            n.end = n.begin + latency
            #logger.debug('## SCHEDULED ## ' + str(n))
            succs = dfg.succs_without_back(n)
            next_candidates = next_candidates.union(succs)
            if longest_latency < n.end:
                longest_latency = n.end
        if next_candidates:
            return self._list_schedule_with_block_bound(dfg, next_candidates, block,
                                                        longest_latency)
        else:
            return longest_latency

    def _node_sched_with_block_bound(self, dfg, node, block):
        preds = dfg.preds_without_back(node)
        preds = [p for p in preds if p.tag.block is block]
        logger.debug('scheduling for ' + str(node))
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, 'DefUse')
            defuse_preds = [p for p in defuse_preds if p.tag.block is block]
            usedef_preds = dfg.preds_typ_without_back(node, 'UseDef')
            usedef_preds = [p for p in usedef_preds if p.tag.block is block]
            seq_preds = dfg.preds_typ_without_back(node, 'Seq')
            seq_preds = [p for p in seq_preds if p.tag.block is block]
            sched_times = []
            if seq_preds:
                latest_node = max(seq_preds, key=lambda p: p.end)
                logger.debug('latest_node of seq_preds ' + str(latest_node))
                if node.tag.is_a([JUMP, CJUMP, MCJUMP]) or has_exclusive_function(node.tag):
                    sched_times.append(latest_node.end)
                else:
                    seq_latency = self.node_seq_latency_map[latest_node]
                    sched_times.append(latest_node.begin + seq_latency)
                    logger.debug('schedtime ' + str(latest_node.begin + seq_latency))
            if defuse_preds:
                latest_node = max(defuse_preds, key=lambda p: p.end)
                logger.debug('latest_node of defuse_preds ' + str(latest_node))
                sched_times.append(latest_node.end)
                logger.debug('schedtime ' + str(latest_node.end))
            if usedef_preds:
                preds = [self._find_latest_alias(dfg, pred) for pred in usedef_preds]
                latest_node = max(preds, key=lambda p: p.begin)
                logger.debug('latest_node(begin) of usedef_preds ' + str(latest_node))
                sched_times.append(latest_node.begin)
                logger.debug('schedtime ' + str(latest_node.begin))
            if not sched_times:
                latest_node = max(preds, key=lambda p: p.begin)
                sched_times.append(latest_node.begin)
            scheduled_time = max(sched_times)
            if scheduled_time < 0:
                scheduled_time = 0
        else:
            # source node
            scheduled_time = 0
        return scheduled_time


class PipelineScheduler(SchedulerImpl):
    def _schedule(self, dfg):
        self._schedule_cycles(dfg)
        self._schedule_ii(dfg)
        conflict_res_table = self._make_conflict_res_table()
        self._schedule_ii_for_conflict(dfg, conflict_res_table)
        self._remove_alias_if_needed(dfg)
        node2state = {}
        block_nodes = self._group_nodes_by_block(dfg)
        longest_latency = 0
        for block, nodes in block_nodes.items():
            latency = self._list_schedule_for_pipeline(dfg,
                                                       nodes,
                                                       conflict_res_table,
                                                       node2state)
            if conflict_res_table:
                latency = self._parallelize_branch_conflict(dfg, conflict_res_table, node2state)
            if longest_latency < latency:
                longest_latency = latency
            self._fill_defuse_gap(dfg, nodes)
        return longest_latency

    def _make_conflict_res_table(self):
        conflict_res_table = defaultdict(list)
        self._extend_conflict_res_table(conflict_res_table, self.res_extractor.mems)
        self._extend_conflict_res_table(conflict_res_table, self.res_extractor.ports)
        return conflict_res_table

    def _extend_conflict_res_table(self, table, node_res_map):
        for node, res in node_res_map.items():
            for r in res:
                table[r].append(node)

    def _schedule_ii(self, dfg):
        initiation_interval = int(dfg.synth_params['ii'])
        if not self.all_paths:
            for path in dfg.trace_all_paths(lambda n: dfg.succs_typ_without_back(n, 'DefUse')):
                self.all_paths.append(path)
        induction_paths = self._find_induction_paths(self.all_paths)
        if initiation_interval < 0:
            latency = self._max_latency(induction_paths)
            dfg.ii = latency if latency > 0 else 1
        else:
            ret, actual = self._adjust_latency(induction_paths, initiation_interval)
            if not ret:
                assert False, 'scheduling of II has failed'
            dfg.ii = actual

    def _find_induction_paths(self, paths):
        induction_paths = []
        for path in paths:
            last_node = path[-1]
            if not last_node.defs:
                continue
            d = last_node.defs[0]
            if not d.is_induction():
                continue
            for i, p in enumerate(path):
                if d in p.uses:
                    induction_paths.append(path[i:])
                    break
        return induction_paths

    def _get_using_resources(self, node):
        res = []
        if node in self.res_extractor.mems:
            res.extend(self.res_extractor.mems[node])
        if node in self.res_extractor.ports:
            res.extend(self.res_extractor.ports[node])
        return res

    def _schedule_ii_for_conflict(self, dfg, conflict_res_table):
        if conflict_res_table.values():
            max_conflict_n = 0
            for nodes in conflict_res_table.values():
                if len(nodes) == 1:
                    continue
                stms = [n.tag for n in nodes]
                for stm in stms:
                    parallel_stms = self.scope.flattened_parallel_hints(stm)
                    if parallel_stms:
                        conflict_n = len(set(stms) - set(parallel_stms))
                        max_conflict_n = max(max_conflict_n, conflict_n)
                    else:
                        max_conflict_n = max(max_conflict_n, len(nodes))
            request_ii = int(dfg.synth_params['ii'])
            if request_ii == -1:
                dfg.ii = max(dfg.ii, max_conflict_n)
            elif request_ii < max_conflict_n:
                fail((self.scope, dfg.region.head.stms[0].lineno),
                     Errors.RULE_INVALID_II, [request_ii, max_conflict_n])

    def _shift_sched_time_for_conflict(self, scheduled_time, n, state_n, conflict_res_table, node2state):
        conflict_node_states = set()
        for r in self._get_using_resources(n):
            if r in conflict_res_table:
                conflict_nodes = list(conflict_res_table[r])
                parallel_stms = self.scope.flattened_parallel_hints(n.tag)
                if parallel_stms:
                    for cn in conflict_nodes[:]:
                        if cn.tag in parallel_stms:
                            conflict_nodes.remove(cn)
                assert n in conflict_nodes
                for cn in conflict_nodes:
                    if cn is not n and cn in node2state:
                        conflict_node_states.add(node2state[cn])
        assert len(conflict_node_states) < state_n
        node_state = scheduled_time % state_n
        while node_state in conflict_node_states:
            scheduled_time += 1
            node_state = scheduled_time % state_n
        return scheduled_time

    def _list_schedule_for_pipeline(self, dfg, nodes, conflict_res_table, node2state):
        next_candidates = set()
        latency = 0
        for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
            scheduled_time = self._node_sched_pipeline(dfg, n)
            _, _, latency = self.node_latency_map[n]
            #detect resource conflict
            # TODO:
            #scheduled_time = self._get_earliest_res_free_time(n, scheduled_time, latency)
            scheduled_time = self._shift_sched_time_for_conflict(scheduled_time, n,
                                                                 dfg.ii,
                                                                 conflict_res_table,
                                                                 node2state)
            n.begin = scheduled_time
            n.end = n.begin + latency
            node_state = n.begin % dfg.ii
            node2state[n] = node_state
            #logger.debug('## SCHEDULED ## ' + str(n))
            #print(node2state[n], n)
            succs = dfg.succs_without_back(n)
            next_candidates = next_candidates.union(succs)
            latency = n.end
        if next_candidates:
            return self._list_schedule_for_pipeline(dfg,
                                                    next_candidates,
                                                    conflict_res_table,
                                                    node2state)
        else:
            return latency

    def _parallelize_branch_conflict(self, dfg, conflict_res_table, node2state):
        next_candidates = set()
        for conflict_nodes in conflict_res_table.values():
            state2node = defaultdict(list)
            for n in conflict_nodes:
                state = node2state[n]
                state2node[state].append(n)
            for n in sorted(conflict_nodes, key=lambda n:n.begin):
                # We search an node that in the same state and also in different stages
                state = node2state[n]
                same_state_nodes = state2node[state]
                diff_stages = [nn.begin for nn in same_state_nodes if nn.begin != n.begin]
                if not diff_stages:
                    # non conflict node
                    continue
                if n.tag in self.scope.parallel_hints:
                    hints_list = self.scope.parallel_hints[n.tag]
                    nearests = []
                    for hints in hints_list:
                        if hints:
                            hint_nodes = set([dfg.find_node(h) for h in hints])
                            xnodes = set(conflict_nodes) & set(hint_nodes)
                            xnodes = sorted(xnodes, key=lambda n:n.begin)
                            nearests.append(xnodes[0])
                    late_nearests = sorted(nearests, key=lambda n:n.begin)[-1]
                    n.begin = late_nearests.begin
                    succs = dfg.succs_without_back(n)
                    next_candidates = next_candidates.union(succs)
        return self._list_schedule_for_pipeline(dfg,
                                                next_candidates,
                                                conflict_res_table,
                                                node2state)

    def _node_sched_pipeline(self, dfg, node):
        preds = dfg.preds_without_back(node)
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, 'DefUse')
            usedef_preds = dfg.preds_typ_without_back(node, 'UseDef')
            seq_preds = dfg.preds_typ_without_back(node, 'Seq')
            sched_times = []
            if seq_preds:
                latest_node = max(seq_preds, key=lambda p: p.end)
                logger.debug('latest_node of seq_preds ' + str(latest_node))
                if node.tag.is_a([JUMP, CJUMP, MCJUMP]) or has_exclusive_function(node.tag):
                    sched_times.append(latest_node.end)
                else:
                    seq_latency = self.node_seq_latency_map[latest_node]
                    sched_times.append(latest_node.begin + seq_latency)
                    logger.debug('schedtime ' + str(latest_node.begin + seq_latency))
            if defuse_preds:
                latest_node = max(defuse_preds, key=lambda p: p.end)
                sched_times.append(latest_node.end)
            if usedef_preds:
                if any([d.is_induction() for d in node.defs]):
                    pass
                else:
                    preds = usedef_preds
                    latest_node = max(preds, key=lambda p: p.end)
                    sched_times.append(latest_node.begin)
            if not sched_times:
                latest_node = max(preds, key=lambda p: p.begin)
                sched_times.append(latest_node.begin)
            scheduled_time = max(sched_times)
            if scheduled_time < 0:
                scheduled_time = 0
        else:
            # source node
            scheduled_time = 0
        return scheduled_time

    def _fill_defuse_gap(self, dfg, nodes):
        for node in reversed(sorted(nodes, key=lambda n: (n.priority, n.stm_index))):
            succs = dfg.succs_without_back(node)
            succs = [s for s in succs if s.begin >= 0]
            if not succs:
                continue
            if self._get_using_resources(node):
                continue
            nearest_node = min(succs, key=lambda p: p.begin)
            sched_time = nearest_node.begin
            if sched_time > node.end:
                gap = sched_time - node.end
                node.begin += gap
                node.end += gap


class ResourceExtractor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.results = []
        self.ops = defaultdict(lambda: defaultdict(int))
        self.mems = defaultdict(list)
        self.ports = defaultdict(list)

    def visit_BINOP(self, ir):
        self.ops[self.current_node][ir.op] += 1
        super().visit_BINOP(ir)

    def visit_CALL(self, ir):
        self.ops[self.current_node][ir.func_scope()] += 1
        func_name = ir.func_scope().name
        if (func_name.startswith('polyphony.io.Port') or
                func_name.startswith('polyphony.io.Queue')):
            inst_ = ir.func.tail()
            self.ports[self.current_node].append(inst_)
        super().visit_CALL(ir)

    def visit_MREF(self, ir):
        if not ir.mem.symbol().typ.get_memnode().can_be_reg():
            self.mems[self.current_node].append(ir.mem.symbol())
        super().visit_MREF(ir)

    def visit_MSTORE(self, ir):
        if not ir.mem.symbol().typ.get_memnode().can_be_reg():
            self.mems[self.current_node].append(ir.mem.symbol())
        super().visit_MSTORE(ir)
