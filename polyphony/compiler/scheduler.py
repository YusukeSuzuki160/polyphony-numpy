import itertools
from collections import defaultdict, deque
from .common import fail, warn
from .dataflow import DFNode
from .errors import Errors, Warnings
from .graph import Graph
from .latency import get_latency
from .irvisitor import IRVisitor
from .ir import *
from .irhelper import has_exclusive_function
from .latency import CALL_MINIMUM_STEP
from .utils import unique
from .scope import Scope
from logging import getLogger
from pyschedule import Scenario, alt, plotters, solvers
from .symbol import Symbol
import re
from matplotlib import pyplot as plt
import time
from .type import Type
from . import utils
from .env import env

logger = getLogger(__name__)

MAX_FUNC_UNIT = 100
MAX_LATENCY = 4


class Scheduler(object):
    def __init__(self):
        self.done_blocks = []

    def schedule(self, scope):
        if scope.is_namespace() or scope.is_class() or scope.is_lib():
            return
        self.scope = scope
        scenarios = []
        res_dicts = {}
        critical_path = []
        critical_path_latency = 0
        for dfg in self.scope.dfgs(bottom_up=True):
            if dfg.parent and dfg.synth_params["scheduling"] == "pipeline":
                scheduler_impl = PipelineScheduler()
            else:
                resource_cond = dfg.synth_params["resource"]
                if scope.is_testbench() or resource_cond == "free":
                    scheduler_impl = BlockBoundedListScheduler()
                else:
                    scheduler_impl = ResourceRestrictedBlockBoundedListScheduler(resource_cond)
                    scope.resource_restrict = True
            scenario = scheduler_impl.schedule(scope, dfg)
            if scenario:
                scenarios += scenario
            scope.append_res_dict(scheduler_impl.res_dict)
            scope.append_bit_dict(scheduler_impl.bit_dict)
            scope.result_vars += scheduler_impl.result_vars
            critival_path_analyzer = CriticalPathAnalyzer(dfg)
            path, latency, _ = critival_path_analyzer.analyze()
            if critical_path_latency <= latency:
                critical_path = path
                critical_path_latency = latency
        if scenarios != []:
            self.plot_result(
                scenarios, "./scheduling/schedule_result" + scope.name + ".png", scope.name
            )
        result_file = open("./scheduling/schedule_result" + self.scope.name + ".txt", "w")
        for res_name, num in scope.res_dict.items():
            result_file.write(res_name + ": " + str(len(num)) + "\n")
        result_file.close()
        critical_path_file = open("./scheduling/critical_path" + self.scope.name + ".txt", "w")
        critical_path_file.write("critical path latency: " + str(critical_path_latency) + "\n")
        for node in critical_path:
            critical_path_file.write(str(node) + "\n")
        critical_path_file.close()

    def plot_result(
        self,
        scenarios,
        img_filename,
        scope_name,
        hide_tasks=[],
        hide_resources=[],
        fig_size=(30, 10),
    ):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        import operator

        resource_height = 1.0
        show_task_labels = True
        vertical_text = False
        color_prec_groups = False
        task_colors = {}
        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        def get_connected_components(edges):
            comps = dict()
            for v, u in edges:
                if v not in comps and u not in comps:
                    comps[v] = v
                    comps[u] = v
                elif v in comps and u not in comps:
                    comps[u] = comps[v]
                elif v not in comps and u in comps:
                    comps[v] = comps[u]
                elif v in comps and u in comps and comps[v] != comps[u]:
                    old_comp = comps[u]
                    for w in comps:
                        if comps[w] == old_comp:
                            comps[w] = comps[v]
            # replace component identifiers by integers startting with 0
            values = list(comps.values())
            comps = {T: values.index(comps[T]) for T in comps}
            return comps

        start_time = 0

        tasks_all = []
        solutions_all = []
        resource_y_positions = {}
        comps_all = {}
        resource_sizes_count = 0
        visible_resources = []
        scenario_end_times = []

        for scenario in scenarios:
            S = scenario
            tasks = [T for T in S.tasks() if T not in hide_tasks]
            tasks = [T for T in tasks if str(T) != "calc_init"]
            tasks_all += tasks
            # get connected components dict for coloring
            # each task is mapping to an integer number which corresponds
            # to its connected component
            edges = [(T, T) for T in tasks]
            if color_prec_groups:
                edges += [
                    (T, T_)
                    for P in set(S.precs_lax()) | set(S.precs_tight())
                    for T in P.tasks()
                    for T_ in P.tasks()
                    if T in tasks and T_ in tasks
                ]
            comps = get_connected_components(edges)
            comps_all.update(comps)

            # color map
            # colors = ["#7EA7D8", "#A1D372", "#EB4845", "#7BCDC8", "#FFF79A"]  # pastel colors
            # # colors = ['red','green','blue','yellow','orange','black','purple'] #basic colors
            # colors += [
            #     [random.random() for i in range(3)] for x in range(len(S.tasks()))
            # ]
            # color_map = {T: colors[comps[T]] for T in comps}
            # replace colors with fixed task colors

            # for T in task_colors:
            #     color_map[T] = task_colors[T]
            hide_tasks_str = [T for T in hide_tasks]
            for T in scenario.tasks():
                if hasattr(T, "plot_color"):
                    if T["plot_color"] is not None:
                        color_map[T] = T["plot_color"]
                    else:
                        hide_tasks_str.append(T)

            solution = S.solution()
            solution = [
                (T, R, x + start_time, y + start_time)
                for (T, R, x, y) in solution
                if T not in hide_tasks_str
            ]
            solutions_all += solution
            start_time = max([y for (T, R, x, y) in solution])
            scenario_end_times.append(start_time)
            for R in scenario.resources():
                if R not in hide_resources and not str(R).startswith("init_sym"):
                    if str(R) not in resource_y_positions:
                        if R.size is not None:
                            resource_size = R.size
                        else:
                            resource_size = 1.0
                        resource_y_positions[str(R)] = resource_sizes_count
                        resource_sizes_count += resource_size
            # if not visible_resources:
            #     raise Exception("ERROR: no resources to plot")

        colors = ["#7EA7D8", "#A1D372", "#EB4845", "#7BCDC8", "#FFF79A"]  # pastel colors
        colors += [[random.random() for i in range(3)] for x in range(len(tasks_all))]
        color_map = {T: colors[comps_all[T]] for T in comps_all}
        for T in task_colors:
            color_map[T] = task_colors[T]
        tasks = tasks_all
        solution = solutions_all
        total_resource_sizes = sum([R.size for R in visible_resources])
        R_ticks = list()
        size_counts = {}

        # for R in visible_resources:
        #     if R.size is not None:
        #         resource_size = R.size
        #     else:
        #         resource_size = 1.0
        #     R_ticks += [str(R.name)] * int(resource_size)
        #     # compute the levels of the tasks on one resource
        #     task_levels = dict()
        #     # get solution on resource sorted according to start time
        #     R_solution = [(T, R_, x, y) for (T, R_, x, y) in solution if R_ == R]
        #     R_solution = sorted(R_solution, key=lambda x: x[2])
        #     # iteratively fill all levels on resource, start with empty fill
        #     level_fill = {i: 0 for i in range(int(resource_size))}
        #     for T, R_, x, y in R_solution:
        #         sorted_levels = sorted(level_fill.items(), key=operator.itemgetter(1, 0))
        #         # get the maximum resource requirement
        #         coeff = max([RA[R] for RA in T.resources_req if R_ in RA])
        #         min_levels = [level for level, fill in sorted_levels[:coeff]]
        #         task_levels[T] = min_levels
        #         for level in min_levels:
        #             level_fill[level] += T.length
        #     # plot solution
        for T, R, x, x_ in solutions_all:
            if T.name == "calc_init":
                continue
            for level in range(int(R.size)):
                y = (
                    resource_sizes_count
                    - 1
                    - (resource_y_positions[str(R)] + level * resource_height)
                )
                ax.add_patch(
                    patches.Rectangle(
                        (x, y),  # (x,y)
                        max(x_ - x, 0.5),  # width
                        resource_height,  # height
                        color=color_map[T],
                        alpha=0.6,
                    )
                )
                if show_task_labels:
                    if vertical_text:
                        text_rotation = 90
                        y_ = y + 0.9 * resource_height
                    else:
                        text_rotation = 0
                        y_ = y + 0.1 * resource_height
                    plt.text(
                        x,
                        y_,
                        str(T.name),
                        fontsize=14,
                        color="black",
                        rotation=text_rotation,
                    )

        prev_end_time = 0
        for i, end_time in enumerate(scenario_end_times):
            plt.axvline(x=end_time, color="gray", linestyle="--", linewidth=1)
            # シナリオ名をプロット
            mid_point = (prev_end_time + end_time) / 2
            plt.text(
                mid_point,
                resource_sizes_count * resource_height,
                scenarios[i].name,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            prev_end_time = end_time
        # format graph
        plt.title(str(scope_name))
        plt.yticks(
            [resource_height * x + resource_height / 2.0 for x in range(resource_sizes_count)],
            [R for R in resource_y_positions][::-1],
        )
        plt.ylim(0, resource_sizes_count * resource_height)
        plt.xlim(0, max([x_ for (T, R, x, x_) in solutions_all if str(R) in resource_y_positions]))
        fig.figsize = fig_size
        plt.savefig(img_filename, dpi=200, bbox_inches="tight")


class SchedulerImpl(object):
    def __init__(self):
        self.res_tables = {}
        self.node_latency_map = {}  # {node:(max, min, actual)}
        self.node_seq_latency_map = {}
        self.all_paths = []
        self.res_extractor = None
        self.res_dict = {}
        self.bit_dict = {}
        self.result_vars = []

    def schedule(self, scope, dfg):
        self.scope = scope
        logger.log(0, "_schedule dfg")
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
        longest_latency, scenario = self._schedule(dfg)
        if longest_latency > CALL_MINIMUM_STEP:
            scope.asap_latency = longest_latency
        else:
            scope.asap_latency = CALL_MINIMUM_STEP
        return scenario

    def _set_priority(self, node, prio, dfg):
        if prio > node.priority:
            node.priority = prio
            logger.debug("update priority ... " + str(node))
            return (dfg.succs_without_back(node), prio + 1)
        return (None, None)

    def _node_sched_default(self, dfg, node):
        preds = dfg.preds_without_back(node)
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, "DefUse")
            usedef_preds = dfg.preds_typ_without_back(node, "UseDef")
            seq_preds = dfg.preds_typ_without_back(node, "Seq")
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
        succs = dfg.succs_typ_without_back(node, "DefUse")
        if not succs:
            return node
        nodes = [self._find_latest_alias(dfg, s) for s in succs]
        latest_node = max(nodes, key=lambda p: p.end)
        return latest_node

    def _is_resource_full(self, res, scheduled_resources):
        # TODO: Limiting resources by scheduler is a future task
        # if isinstance(res, str):
        #    return len(scheduled_resources) >= MAX_FUNC_UNIT
        # elif isinstance(res, Scope):
        #    return len(scheduled_resources) >= MAX_FUNC_UNIT
        return 0

    def _str_res(self, res):
        if isinstance(res, str):
            return res
        elif isinstance(res, Scope):
            return res.name

    def _get_earliest_res_free_time(self, node, time, latency):
        resources = self.res_extractor.ops[node].keys()
        # TODO operator chaining?
        # logger.debug(node)
        # logger.debug(resources)
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
                # already scheduled
                return time

            while self._is_resource_full(res, scheduled_resources):
                logger.debug(
                    "!!! resource {}'s slot '{}' is full !!!".format(self._str_res(res), time)
                )
                assert False, "Rescheduling due to lack of resources is not supported yet"
                time += 1
                scheduled_resources = table[time]

            node.instance_num = len(scheduled_resources)
            # logger.debug("{} is scheduled to {}, instance_num {}".
            #             format(node, time, node.instance_num))

            # fill scheduled_resources table
            n = latency if latency != 0 else 1
            for i in range(n):
                scheduled_resources = table[time + i]
                scheduled_resources.append(node)
        return time

    def _calc_latency(self, dfg):
        is_minimum = dfg.synth_params["cycle"] == "minimum"
        for node in dfg.get_priority_ordered_nodes():
            def_l, seq_l = get_latency(node.tag)
            if def_l == 0:
                if is_minimum:
                    self.node_latency_map[node] = (0, 0, 0)
                else:
                    if node.tag.is_a([MOVE, PHIBase]):
                        var = (
                            node.tag.dst.symbol() if node.tag.is_a(MOVE) else node.tag.var.symbol()
                        )
                        # if var.is_condition():
                        #     self.node_latency_map[node] = (0, 0, 0)
                        # else:
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
        for path in dfg.trace_all_paths(lambda n: dfg.succs_typ_without_back(n, "DefUse")):
            self.all_paths.append(path)
        ret, actual = self._adjust_latency(self.all_paths, expected)
        if not ret:
            assert False, "scheduling has failed. the cycle must be greater equal {}".format(actual)

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
                        d.del_tag("alias")

    def _group_nodes_by_block(self, dfg):
        block_nodes = defaultdict(list)
        for node in dfg.get_priority_ordered_nodes():
            block_nodes[node.tag.block].append(node)
        return block_nodes

    def _schedule_cycles(self, dfg):
        self._calc_latency(dfg)
        synth_cycle = dfg.synth_params["cycle"]
        if synth_cycle == "any" or synth_cycle == "minimum":
            pass
        elif synth_cycle.startswith("less:"):
            extected_latency = int(synth_cycle[len("less:") :])
            self._try_adjust_latency(dfg, extected_latency)
        elif synth_cycle.startswith("greater:"):
            assert False, "Not Implement Yet"
        else:
            assert False


class BlockBoundedListScheduler(SchedulerImpl):
    def _schedule(self, dfg):
        self._schedule_cycles(dfg)
        self._remove_alias_if_needed(dfg)

        block_nodes = self._group_nodes_by_block(dfg)
        longest_latency = 0
        for block, nodes in block_nodes.items():
            # latency = self._list_schedule(dfg, nodes)
            print("Block: " + block.name)
            for node in nodes:
                print(node)
            latency = self._list_schedule_with_block_bound(dfg, nodes, block, 0)
            if longest_latency < latency:
                longest_latency = latency
            critical_path_analyzer = CriticalPathAnalyzer(dfg)
            critical_path, critical_path_latency, split_points = critical_path_analyzer.analyze()
        print("\ndfg before split")
        print(dfg)
        while critical_path_latency > MAX_LATENCY:
            stage = 0
            split_point = split_points[stage]
            for i, node in enumerate(critical_path):
                if i == split_point:
                    if node.begin == node.end:
                        node.begin += stage
                        node.end += stage + 1
                        self.node_latency_map[node] = (1, 0, 1)
                    else:
                        node.begin += stage
                        node.end += stage
                    stage += 1
                    if stage == len(split_points):
                        split_points = len(critical_path)
                    else:
                        split_point = split_points[stage]
                else:
                    node.begin += stage
                    node.end += stage
            succs = []
            for node in critical_path:
                succ = (
                    dfg.succs_typ_without_back(node, "DefUse")
                    + dfg.preds_typ_without_back(node, "UseDef")
                    + dfg.succs_typ_without_back(node, "Seq")
                )
                succs += succ
            succs = utils.unique(succs)
            visited = critical_path
            while succs:
                nodes = succs
                succs = []
                for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
                    if n in visited:
                        continue
                    visited.append(n)
                    succ = (
                        dfg.succs_typ_without_back(n, "DefUse")
                        + dfg.preds_typ_without_back(n, "UseDef")
                        + dfg.succs_typ_without_back(n, "Seq")
                    )
                    succs += succ
                    n.begin += stage
                    n.end += stage
                succs = utils.unique(succs)
            self._remove_alias_if_needed(dfg)
            critical_path_analyzer = CriticalPathAnalyzer(dfg)
            critical_path, critical_path_latency, split_points = critical_path_analyzer.analyze()
        print("\ndfg after split")
        print(dfg)
        return longest_latency, None

    def _list_schedule(self, dfg, nodes):
        while True:
            next_candidates = set()
            latency = 0
            for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
                scheduled_time = self._node_sched(dfg, n)
                latency = get_latency(n.tag)
                # detect resource conflict
                scheduled_time = self._get_earliest_res_free_time(n, scheduled_time, latency)
                n.begin = scheduled_time
                n.end = n.begin + latency
                # logger.debug('## SCHEDULED ## ' + str(n))
                succs = dfg.succs_without_back(n)
                next_candidates = next_candidates.union(succs)
                latency = n.end
            if next_candidates:
                nodes = next_candidates
            else:
                break
        return latency

    def _list_schedule_with_block_bound(self, dfg, nodes, block, longest_latency, excepts = []):
        while True:
            next_candidates = set()
            for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
                if n.tag.block is not block:
                    continue
                if n in excepts:
                    continue
                scheduled_time = self._node_sched_with_block_bound(dfg, n, block)
                _, _, latency = self.node_latency_map[n]
                # detect resource conflict
                scheduled_time = self._get_earliest_res_free_time(n, scheduled_time, latency)
                n.begin = scheduled_time
                n.end = n.begin + latency
                # logger.debug('## SCHEDULED ## ' + str(n))
                succs = dfg.succs_without_back(n)
                next_candidates = next_candidates.union(succs)
                if longest_latency < n.end:
                    longest_latency = n.end
            if next_candidates:
                nodes = next_candidates
            else:
                break
        return longest_latency

    def _node_sched_with_block_bound(self, dfg, node, block):
        preds = dfg.preds_without_back(node)
        preds = [p for p in preds if p.tag.block is block]
        logger.debug("scheduling for " + str(node))
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, "DefUse")
            defuse_preds = [p for p in defuse_preds if p.tag.block is block]
            usedef_preds = dfg.preds_typ_without_back(node, "UseDef")
            usedef_preds = [p for p in usedef_preds if p.tag.block is block]
            seq_preds = dfg.preds_typ_without_back(node, "Seq")
            seq_preds = [p for p in seq_preds if p.tag.block is block]
            sched_times = []
            if seq_preds:
                if node.tag.is_a([JUMP, CJUMP, MCJUMP]) or has_exclusive_function(node.tag):
                    latest_node = max(seq_preds, key=lambda p: p.end)
                    sched_time = latest_node.end
                else:
                    latest_node = max(seq_preds, key=lambda p: (p.begin, p.end))
                    seq_latency = self.node_seq_latency_map[latest_node]
                    sched_time = latest_node.begin + seq_latency
                sched_times.append(sched_time)
                logger.debug("latest_node of seq_preds " + str(latest_node))
                logger.debug("schedtime " + str(sched_time))
            if defuse_preds:
                latest_node = max(defuse_preds, key=lambda p: p.end)
                logger.debug("latest_node of defuse_preds " + str(latest_node))
                sched_times.append(latest_node.end)
                logger.debug("schedtime " + str(latest_node.end))
            if usedef_preds:
                preds = [self._find_latest_alias(dfg, pred) for pred in usedef_preds]
                latest_node = max(preds, key=lambda p: p.begin)
                logger.debug("latest_node(begin) of usedef_preds " + str(latest_node))
                sched_times.append(latest_node.begin)
                logger.debug("schedtime " + str(latest_node.begin))
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


class ResourceRestrictedBlockBoundedListScheduler(SchedulerImpl):
    def __init__(self, resources: dict[str, int]) -> None:
        super().__init__()
        self.resources = resources
        self.scheduler = None
        self.schedule_result = {}
        self.phonys = []
        self.hide_tasks = []

    def _schedule(self, dfg):
        self._calc_latency(dfg)
        self._remove_alias_if_needed(dfg)
        for path in dfg.trace_all_paths(lambda n: dfg.succs_typ_without_back(n, "DefUse")):
            self.all_paths.append(path)
        cycle = dfg.synth_params["cycle"]
        if cycle == "minimum" or cycle == "any":
            clock_limit = self._max_latency(self.all_paths) + 1
        elif cycle.startswith("less:"):
            extected_latency = int(cycle[len("less:") :])
            clock_limit = extected_latency
        block_nodes = self._group_nodes_by_block(dfg)
        total_latency = 0
        scenarios = []
        for block, node in block_nodes.items():
            max_latency = 0
            self.scheduler = Scenario(block.name, horizon=clock_limit)
            max_res = self.res_extractor.calc_resources_in_block(block)
            max_res = {k if isinstance(k, str) else k.name: v for k, v in max_res.items()}
            resources = {}
            for res, num in max_res.items():
                res_name = res if isinstance(res, str) else res.name
                if res_name in self.resources:
                    num = self.resources[res]
                match res_name:
                    case "Add" | "Sub" | "Mult" | "Div" | "FloorDiv" | "Mod" | "BitOr" | "BitAnd" | "BitXor" | "And" | "Or" | "Eq" | "NotEq" | "Lt" | "LtE" | "Gt" | "GtE":
                        res_name = res_name + "er"
                    case "RShift" | "LShift":
                        continue
                    case _:
                        res_name = "func_" + res_name
                resources[res] = self.scheduler.Resources(res_name, num=num)
            if resources == {}:
                print("\nblock: " + block.name)
                print("node num: ", 0)
                for n in node:
                    _, _, latency = self.node_latency_map[n]
                    sched_time = self._node_sched_with_block_bound(dfg, n, block)
                    n.begin = sched_time
                    n.end = n.begin + latency
                self._remove_alias_if_needed(dfg)
                continue
            resources["start"] = self.scheduler.Resources("init_sym", num=1)
            tasks = self._add_nodes_to_scheduler(dfg, node, block, resources)
            if "solver" in dfg.synth_params:
                solver = dfg.synth_params["solver"]
            else:
                solver = env.config.scheduler_params["solver"]
            msg = env.config.scheduler_params["msg"]
            start = time.time()
            if solver == "list scheduling":
                is_success = solvers.listsched.solve(self.scheduler, msg=msg)
            elif solver == "mip":
                is_success = solvers.mip.solve(self.scheduler, msg=msg)
            elif solver == "cp":
                is_success = solvers.cpoptimizer.solve(self.scheduler, msg=msg)
            elif solver == "ortools":
                is_success = solvers.ortools.solve(self.scheduler, msg=msg)
            elif solver == "cplex":
                is_success = solvers.mip.solve(self.scheduler, msg=msg, kind="CPLEX")
            elif solver == "glpk":
                is_success = solvers.mip.solve(self.scheduler, msg=msg, kind="GLPK")
            elif solver == "cbc":
                is_success = solvers.mip.solve(self.scheduler, msg=msg, kind="CBC")
            elif solver == "scip":
                is_success = solvers.mip.solve(self.scheduler, msg=msg, kind="SCIP")
            elif solver == "gurobi":
                is_success = solvers.mip.solve(self.scheduler, msg=msg, kind="GUROBI")
            else:
                assert False, "unknown solver: {}".format(solver)
            if not is_success:
                raise Exception("scheduling failed")
            end = time.time()
            print("scheduling time: {}[s]".format(end - start))
            results = self.get_result()
            self.schedule_result[block] = results
            save_file_name = "./scheduling/schedule_result_" + block.name + ".png"
            # hide_tasks = [v for k, v in tasks.items() if str(v).startswith("other")] + self.hide_tasks
            # hide_resources = self.phonys
            hide_tasks = []
            hide_resources = []
            # plotters.matplotlib.plot(
            #     self.scheduler,
            #     img_filename=save_file_name,
            #     hide_tasks=hide_tasks,
            #     hide_resources=hide_resources,
            # )
            tasks = {str(v): k for k, v in tasks.items()}
            # for task in self.schedule_result[block]:
            #     task_name = task[0]
            #     res = task[1]
            #     begin = int(task[2])
            #     end = int(task[3])
            #     if max_latency < end:
            #         max_latency = end
            #     node = tasks[task_name]
            #     self._apply_schedule_result(dfg, node, begin, end)
            #     self._add_node_to_res_table(node, begin, end, res)
            #     self._remove_alias_if_needed(dfg)
            visited = []
            up_priorities = []
            using_resources = {}
            first = node[0]
            min_prio = first.priority
            for n in node:
                if n.priority < min_prio:
                    min_prio = n.priority
                    first = n
            # first_offset = self._node_sched_with_block_bound(dfg, first, block)
            first_offset = 0
            for task in results:
                if task[0] == "calc_init":
                    continue
                res = task[1]
                resource_name = re.match(r"^(.*?)(\d+)$", res)
                if resource_name:
                    res_num = int(resource_name.group(2))
                    resource_name = resource_name.group(1)
                else:
                    continue
                n = tasks[task[0]]
                if isinstance(n.tag.src.left, TEMP):
                    left_width = n.tag.src.left.sym.typ.get_width()
                else:
                    left_width = env.config.default_int_width
                if isinstance(n.tag.src.right, TEMP):
                    right_width = n.tag.src.right.sym.typ.get_width()
                else:
                    right_width = env.config.default_int_width
                op = n.tag.src.op
                left_name = str(op) + "_" + str(res_num) + "_left"
                right_name = str(op) + "_" + str(res_num) + "_right"
                if left_name not in self.bit_dict.keys():
                    self.bit_dict[left_name] = left_width
                else:
                    self.bit_dict[left_name] = max(self.bit_dict[left_name], left_width)
                if right_name not in self.bit_dict.keys():
                    self.bit_dict[right_name] = right_width
                else:
                    self.bit_dict[right_name] = max(self.bit_dict[right_name], right_width)
            for task in results:
                if task[0] == "calc_init":
                    continue
                n = tasks[task[0]]
                begin = int(task[2]) + first_offset
                end = int(task[3]) + first_offset
                res = task[1]
                if max_latency < end:
                    max_latency = end
                priority = n.priority
                if priority not in up_priorities:
                    for n2 in node:
                        if n2.priority > priority:
                            n2.priority += 1
                    up_priorities.append(priority)
                self._add_node_to_res_table(n, begin + 1, end + 1, res, using_resources)
                self._apply_schedule_result(dfg, n, begin, end, tasks, visited)
            # self._remove_alias_if_needed(dfg)
            for n in sorted(node, key=lambda n: (n.priority, n.stm_index)):
                if n not in visited:
                    visited.append(n)
                    latency, _, _ = self.node_latency_map[n]
                    scheduled_time = self._node_sched_with_block_bound(dfg, n, block)
                    n.begin = scheduled_time
                    n.end = n.begin + latency
            # self._remove_alias_if_needed(dfg)
            total_latency += max_latency
            # print(using_resources)
            # print("resources", resources.keys())
            # print(dfg)
            for res_name, res in resources.items():
                res_name = res_name if isinstance(res_name, str) else res_name.name
                if res_name == "start":
                    continue
                num = using_resources[res_name]
                if res_name in self.res_dict.keys():
                    self.res_dict[res_name] += num
                    self.res_dict[res_name] = list(set(self.res_dict[res_name]))
                else:
                    self.res_dict[res_name] = num
            self.phonys = []
            self.hide_tasks = []
            scenarios.append(self.scheduler)
        return max_latency, scenarios

    def _add_nodes_to_scheduler(self, dfg, nodes, block, resources) -> dict[DFNode, Scenario.Task]:
        print("\nblock: " + block.name)
        tasks = {}
        node_dict = {}
        node_num = 0
        is_first = True
        max_priority = max([n.priority for n in nodes])
        start = self.scheduler.Task("calc_init", length=0, delay_cost=1)
        start += resources["start"]
        for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
            if n.tag.block is not block:
                continue
            _, _, latency = self.node_latency_map[n]
            # if latency == 0:
            #     continue
            if n.tag.is_a(MOVE):
                if n.tag.src.is_a([BINOP]) and n.tag.src.op in (
                    "Add",
                    "Sub",
                    "Mult",
                    "Div",
                    "FloorDiv",
                    "Mod",
                    "BitOr",
                    "BitAnd",
                    "BitXor",
                    "And",
                    "Or",
                    "Eq",
                    "NotEq",
                    "Lt",
                    "LtE",
                    "Gt",
                    "GtE",
                ):
                    op = n.tag.src.op
                    task = self.scheduler.Task(
                        op + str(node_num), length=latency, delay_cost=max_priority - n.priority + 1
                    )
                    task += alt(resources[op])
                    # result_task = self.scheduler.Task(
                    #     "result" + str(node_num), length=1, delay_cost=1
                    # )
                    # phony = self.scheduler.Resources(
                    #     "phony_" + str(phony_num) + "_", num=1
                    # )
                    # phony_num += 1
                    # result_task += phony
                    # self.phonys.append(phony[0])
                    # self.scheduler += task < result_task
                    # task = (task, result_task)
                    # self.hide_tasks.append(result_task)
                elif n.tag.src.is_a(CALL):
                    func = n.tag.src.func_scope()
                    task = self.scheduler.Task(
                        func.name + str(node_num),
                        length=latency,
                        delay_cost=max_priority - n.priority + 1,
                    )
                    task += alt(resources[func.name])
                else:
                    # task = self.scheduler.Task(
                    #     "other" + str(node_num), length=latency, delay_cost=1
                    # )
                    # phony = self.scheduler.Resources(
                    #     "phony_" + str(phony_num) + "_", num=1
                    # )
                    # phony_num += 1
                    # task += phony
                    # self.phonys.append(phony[0])
                    continue
            elif n.tag.is_a(EXPR):
                if n.tag.exp.is_a(CALL):
                    task = self.scheduler.Task(
                        n.tag.exp.func_scope().name + str(node_num),
                        length=latency,
                        delay_cost=max_priority - n.priority + 1,
                    )
                    func = n.tag.exp.func_scope()
                    task += alt(resources[func.name])
                    if is_first:
                        first_prio = n.priority
                        is_first = False
                else:
                    # task = self.scheduler.Task(
                    #     "other" + str(node_num), length=latency, delay_cost=1
                    # )
                    # phony = self.scheduler.Resources(
                    #     "phony_" + str(phony_num) + "_", num=1
                    # )
                    # phony_num += 1
                    # task += phony
                    # self.phonys.append(phony[0])
                    continue
            else:
                # task = self.scheduler.Task(
                #     "other" + str(node_num), length=latency, delay_cost=1
                # )
                # phony = self.scheduler.Resources("phony_" + str(phony_num) + "_", num=1)
                # phony_num += 1
                # task += phony
                # self.phonys.append(phony[0])
                continue
            tasks[n] = task
            node_dict[task] = n
            node_num += 1
        for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
            _, _, latency = self.node_latency_map[n]
            sched_time = self._node_sched_with_block_bound(dfg, n, block)
            n.begin = sched_time
            n.end = n.begin + latency
        for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
            if n.tag.block is not block:
                continue
            if n not in tasks.keys():
                continue
            visited = [n]
            start_time = self._node_sched_with_block_bound(dfg, n, block)
            if start_time == 0:
                self.scheduler += start < tasks[n]
                # print("start < " + tasks[n].name)
            else:
                self.scheduler += start + start_time < tasks[n]
                # print("start + " + str(start_time) + " < " + tasks[n].name)
            succs = (
                dfg.succs_typ_without_back(n, "DefUse")
                + dfg.preds_typ_without_back(n, "UseDef")
                + dfg.succs_typ_without_back(n, "Seq")
            )
            succs = utils.unique(succs)
            task_pair = {}
            succ_stack = []
            for succ in succs:
                _, _, latency = self.node_latency_map[succ]
                succ_stack.append((succ, latency))
            while succ_stack:
                next_succs = []
                succ, distance = succ_stack.pop()
                if succ.priority <= n.priority:
                    continue
                if succ in visited:
                    continue
                if succ not in tasks.keys():
                    visited.append(succ)
                    next_succs += dfg.succs_typ_without_back(succ, "DefUse")
                    next_succs += dfg.preds_typ_without_back(succ, "UseDef")
                    next_succs += dfg.succs_typ_without_back(succ, "Seq")
                    next_succs = utils.unique(next_succs)
                    if next_succs:
                        for next_succ in next_succs:
                            _, _, latency = self.node_latency_map[next_succ]
                            next_succ_stack = (next_succ, latency + distance)
                            succ_stack.append(next_succ_stack)
                else:
                    succ_task = tasks[succ]
                    pair = (tasks[n], succ_task)
                    if pair not in task_pair.keys():
                        task_pair[pair] = distance
                    else:
                        task_pair[pair] = min(task_pair[pair], distance)
                        distance = task_pair[pair]
            for pair, distance in task_pair.items():
                self.scheduler += pair[0] + distance < pair[1]
                # print(pair[0].name + " + " + str(distance) + " < " + pair[1].name)
        # print("stms before: ")
        # for i, n in enumerate(sorted(nodes, key=lambda n: (n.priority, n.stm_index))):
        #     if n in tasks.keys():
        #         print(str(i) + ": " + str(n), ": ", tasks[n])
        #     else:
        #         print(str(i) + ": " + str(n))
        print("node num: " + str(node_num))
        tasks = {k: v[0] if isinstance(v, tuple) else v for k, v in tasks.items()}
        return tasks

    def _add_node_to_res_table(self, node, begin, end, res, using):
        resource_name = re.match(r"^(.*?)(\d+)$", res)
        if resource_name:
            instance_num = int(resource_name.group(2))
            resource_name = resource_name.group(1)
        else:
            return
        if resource_name.endswith("er"):
            resource_name = resource_name[:-2]
        elif resource_name.startswith("func_"):
            resource_name = resource_name[5:]
        node.instance_num = instance_num
        if resource_name not in using:
            using[resource_name] = [instance_num]
        else:
            if instance_num not in using[resource_name]:
                using[resource_name].append(instance_num)
        if resource_name not in self.res_tables:
            table = defaultdict(list)
            self.res_tables[resource_name] = table
        else:
            table = self.res_tables[resource_name]
        for i in range(begin, end):
            table[i].append(node)

    def _apply_schedule_result(self, dfg, node, begin, end, tasks, visited):
        visited.append(node)
        if node.tag.is_a(MOVE) and node.tag.src.is_a([BINOP, UNOP, RELOP]):
            op = node.tag.src.op
            stm_index = node.stm_index
            scope = dfg.scope
            block = node.tag.block
            priority = node.priority
            lineno = node.tag.loc.lineno
            filename = node.tag.loc.filename
            left_name = str(op) + "_" + str(node.instance_num) + "_left"
            right_name = str(op) + "_" + str(node.instance_num) + "_right"
            int_type = Type.int()
            default_tags = {"induction"}
            if isinstance(node.tag.src.left, TEMP):
                left_typ = node.tag.src.left.sym.typ
                tags = node.tag.src.left.sym.tags
                left_init = TEMP(Symbol(left_name, tags=tags, scope=scope, typ=left_typ), Ctx.STORE)
            elif isinstance(node.tag.src.left, CONST):
                left_init = TEMP(
                    Symbol(left_name, tags=default_tags, scope=scope, typ=int_type), Ctx.STORE
                )
            elif isinstance(node.tag.src.left, UNOP):
                left_typ = node.tag.src.left.exp.sym.typ
                tags = node.tag.src.left.exp.sym.tags
                left_init = TEMP(Symbol(left_name, tags=tags, scope=scope, typ=left_typ), Ctx.STORE)
            if isinstance(node.tag.src.right, TEMP):
                right_typ = node.tag.src.right.sym.typ
                tags = node.tag.src.right.sym.tags
                right_init = TEMP(
                    Symbol(right_name, tags=tags, scope=scope, typ=right_typ), Ctx.STORE
                )
            elif isinstance(node.tag.src.right, CONST):
                right_init = TEMP(
                    Symbol(right_name, tags=default_tags, scope=scope, typ=int_type), Ctx.STORE
                )
            elif isinstance(node.tag.src.right, UNOP):
                right_typ = node.tag.src.right.exp.sym.typ
                tags = node.tag.src.right.exp.sym.tags
                right_init = TEMP(
                    Symbol(right_name, tags=tags, scope=scope, typ=right_typ), Ctx.STORE
                )
            node_succ = dfg.succ_edges[node]
            node_pred = dfg.pred_edges[node]
            removes = []
            for (n1, n2), (typ, _) in dfg.edges.items():
                if n1 == node:
                    removes.append((n1, n2, typ))
                elif n2 == node:
                    removes.append((n1, n2, typ))
            if removes != []:
                for n1, n2, _ in removes:
                    dfg.remove_edge(n1, n2)
            dfg.remove_node(node)
            if node in dfg.src_nodes:
                dfg.src_nodes.remove(node)
            # if op in ("Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "BitOr", "BitAnd", "BitXor"):
            #     calc_node = BINOP(op, left, right)
            # elif op in ("And", "Or", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"):
            #     calc_node = RELOP(op, left, right)
            calc_result_name = str(op) + "_" + str(node.instance_num) + "_result"
            dst_type = node.tag.dst.sym.typ
            calc_result_sym = Symbol(calc_result_name, tags=default_tags, scope=scope, typ=dst_type)
            # calc_result_sym.add_tag("alias")
            calc_node = TEMP(calc_result_sym, Ctx.LOAD)
            # node.tag.dst.sym.add_tag("alias")
            calc_node = MOVE(node.tag.dst, calc_node, loc=Loc(filename, lineno))
            left_init.sym.del_tag("alias")
            right_init.sym.del_tag("alias")
            init_left = MOVE(left_init, node.tag.src.left, loc=Loc(filename, lineno))
            init_right = MOVE(right_init, node.tag.src.right, loc=Loc(filename, lineno))
            dst_name = node.tag.dst.sym.name
            if dst_name.startswith("@"):
                dst_name = dst_name[1:]
            dst_name = dst_name.replace("#", "")
            self.result_vars.append(dst_name)
            # print("\nbefore")
            # print(stm_index)
            # print(str(node))
            # for i, stm in enumerate(block.stms):
            #     print(str(i) + ": " + str(stm))
            block.stms[stm_index] = calc_node
            block.stms.insert(stm_index, init_left)
            block.stms.insert(stm_index, init_right)
            # print("\nafter")
            # for i, stm in enumerate(block.stms):
            #     print(str(i) + ": " + str(stm))
            calc_node.block = block
            init_left.block = block
            init_right.block = block
            # print("\nnode: ", node)
            # print("calc_node: " + str(calc_node))
            # print("init_left: " + str(init_left))
            # print("init_right: " + str(init_right))
            dfg.update_stm_index()
            calc_node = dfg.add_stm_node(calc_node)
            init_left = dfg.add_stm_node(init_left)
            init_right = dfg.add_stm_node(init_right)
            dfg.src_nodes.add(init_left)
            dfg.src_nodes.add(init_right)
            dfg.src_nodes.add(calc_node)
            init_left.begin = begin
            init_left.end = begin + 1
            init_right.begin = begin
            init_right.end = begin + 1
            calc_node.begin = begin + 1
            calc_node.end = begin + 2
            init_left.priority = priority
            init_right.priority = priority
            calc_node.priority = priority + 1
            calc_node.instance_num = node.instance_num
            # end = end + 1
            dfg.add_defuse_edge(init_left, calc_node)
            dfg.add_defuse_edge(init_right, calc_node)
            self.node_latency_map[init_left] = (1, 1, 1)
            self.node_latency_map[init_right] = (1, 1, 1)
            self.node_latency_map[calc_node] = (0, 0, 0)
            if removes != []:
                for n1, n2, typ in removes:
                    if n1 == node:
                        match typ:
                            case "DefUse":
                                dfg.add_defuse_edge(calc_node, n2)
                            case "UseDef":
                                dfg.add_usedef_edge(init_left, n2)
                                dfg.add_usedef_edge(init_right, n2)
                    elif n2 == node:
                        match typ:
                            case "DefUse":
                                dfg.add_defuse_edge(n1, init_left)
                                dfg.add_defuse_edge(n1, init_right)
                            case "UseDef":
                                dfg.add_usedef_edge(n1, calc_node)
            new_succ_calc = defaultdict(set)
            new_pred_calc = defaultdict(set)
            new_succ_init = defaultdict(set)
            new_pred_init = defaultdict(set)
            for succ in node_succ:
                typ = succ[3]
                if typ == "DefUse" or typ == "Seq":
                    new_succ_calc.add(succ)
                elif typ == "UseDef":
                    new_succ_init.add(succ)
            for pred in node_pred:
                typ = pred[3]
                if typ == "DefUse" or typ == "Seq":
                    new_pred_init.add(pred)
                elif typ == "UseDef":
                    new_pred_calc.add(pred)
            for succ in new_succ_calc:
                dfg.succ_edges[calc_node].add(succ)
            for pred in new_pred_calc:
                dfg.pred_edges[calc_node].add(pred)
            for succ in new_succ_init:
                dfg.succ_edges[init_left].add(succ)
                dfg.succ_edges[init_right].add(succ)
            for pred in new_pred_init:
                dfg.pred_edges[init_left].add(pred)
                dfg.pred_edges[init_right].add(pred)
            dfg.update_stm_index()
        else:
            node.begin = begin
            node.end = end

    def _node_sched_with_block_bound(self, dfg, node, block):
        preds = dfg.preds_without_back(node)
        preds = [p for p in preds if p.tag.block is block]
        logger.debug("scheduling for " + str(node))
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, "DefUse")
            defuse_preds = [p for p in defuse_preds if p.tag.block is block]
            usedef_preds = dfg.preds_typ_without_back(node, "UseDef")
            usedef_preds = [p for p in usedef_preds if p.tag.block is block]
            seq_preds = dfg.preds_typ_without_back(node, "Seq")
            seq_preds = [p for p in seq_preds if p.tag.block is block]
            sched_times = []
            if seq_preds:
                if node.tag.is_a([JUMP, CJUMP, MCJUMP]) or has_exclusive_function(node.tag):
                    latest_node = max(seq_preds, key=lambda p: p.end)
                    sched_time = latest_node.end
                else:
                    latest_node = max(seq_preds, key=lambda p: (p.begin, p.end))
                    seq_latency = self.node_seq_latency_map[latest_node]
                    sched_time = latest_node.begin + seq_latency
                sched_times.append(sched_time)
                logger.debug("latest_node of seq_preds " + str(latest_node))
                logger.debug("schedtime " + str(sched_time))
            if defuse_preds:
                latest_node = max(defuse_preds, key=lambda p: p.end)
                logger.debug("latest_node of defuse_preds " + str(latest_node))
                sched_times.append(latest_node.end)
                logger.debug("schedtime " + str(latest_node.end))
            if usedef_preds:
                preds = [self._find_latest_alias(dfg, pred) for pred in usedef_preds]
                latest_node = max(preds, key=lambda p: p.begin)
                logger.debug("latest_node(begin) of usedef_preds " + str(latest_node))
                sched_times.append(latest_node.begin)
                logger.debug("schedtime " + str(latest_node.begin))
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

    def get_result(self):
        solution = []

        for i in range(len(self.scheduler.solution())):
            solution.append([])
            for j in range(len(self.scheduler.solution()[i])):
                solution[i].append(str(self.scheduler.solution()[i][j]))

        return solution


class PipelineScheduler(SchedulerImpl):
    def _schedule(self, dfg):
        self._schedule_cycles(dfg)
        self._schedule_ii(dfg)
        self._remove_alias_if_needed(dfg)
        self.d2c = {}
        block_nodes = self._group_nodes_by_block(dfg)
        longest_latency = 0
        for block, nodes in block_nodes.items():
            latency = self._list_schedule_for_pipeline(dfg, nodes, 0)
            conflict_res_table = self._make_conflict_res_table(nodes)
            if conflict_res_table:
                logger.debug("before rescheduling")
                for n in dfg.get_scheduled_nodes():
                    logger.debug(n)
                latency = self._reschedule_for_conflict(dfg, conflict_res_table, latency)
            if longest_latency < latency:
                longest_latency = latency
            self._fill_defuse_gap(dfg, nodes)
        logger.debug("II = " + str(dfg.ii))
        return longest_latency

    def _make_conflict_res_table(self, nodes):
        conflict_res_table = defaultdict(list)
        self._extend_conflict_res_table(conflict_res_table, nodes, self.res_extractor.mems)
        self._extend_conflict_res_table(conflict_res_table, nodes, self.res_extractor.ports)
        self._extend_conflict_res_table(conflict_res_table, nodes, self.res_extractor.regarrays)
        return conflict_res_table

    def _extend_conflict_res_table(self, table, target_nodes, node_res_map):
        for node, res in node_res_map.items():
            if node not in target_nodes:
                continue
            for r in res:
                table[r].append(node)

    def max_cnode_num(self, cgraph):
        max_cnode = defaultdict(int)
        for cnode in cgraph.get_nodes():
            max_cnode[cnode.res] += 1
        if max_cnode:
            return max(list(max_cnode.values()))
        else:
            return 0

    def _schedule_ii(self, dfg):
        initiation_interval = int(dfg.synth_params["ii"])
        if not self.all_paths:
            for path in dfg.trace_all_paths(lambda n: dfg.succs_typ_without_back(n, "DefUse")):
                self.all_paths.append(path)
        induction_paths = self._find_induction_paths(self.all_paths)
        if initiation_interval < 0:
            latency = self._max_latency(induction_paths)
            dfg.ii = latency if latency > 0 else 1
        else:
            ret, actual = self._adjust_latency(induction_paths, initiation_interval)
            if not ret:
                assert False, "scheduling of II has failed"
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
        if node in self.res_extractor.regarrays:
            res.extend(self.res_extractor.regarrays[node])
        return res

    def find_cnode(self, cgraph, stm):
        if cgraph is None:
            return None
        for cnode in cgraph.get_nodes():
            if isinstance(cnode, ConflictNode):
                if stm in cnode.items:
                    return cnode
            else:
                if stm is cnode:
                    return cnode
        return None

    def _list_schedule_for_pipeline(self, dfg, nodes, longest_latency):
        while True:
            next_candidates = set()
            for n in sorted(nodes, key=lambda n: (n.priority, n.stm_index)):
                scheduled_time = self._node_sched_pipeline(dfg, n)
                _, _, latency = self.node_latency_map[n]
                # detect resource conflict
                # TODO:
                # scheduled_time = self._get_earliest_res_free_time(n, scheduled_time, latency)
                if scheduled_time > n.begin:
                    n.begin = scheduled_time
                    if n in self.d2c:
                        cnode = self.d2c[n]
                        for dnode in cnode.items:
                            if dnode is n:
                                continue
                            dnode.begin = n.begin
                            next_candidates.add(dnode)
                n.end = n.begin + latency
                # logger.debug('## SCHEDULED ## ' + str(n))
                succs = dfg.succs_without_back(n)
                next_candidates = next_candidates.union(succs)
                if longest_latency < n.end:
                    longest_latency = n.end
            if next_candidates:
                nodes = next_candidates
            else:
                break
        return longest_latency

    def _reschedule_for_conflict(self, dfg, conflict_res_table, longest_latency):
        self.cgraph = ConflictGraphBuilder(self.scope, dfg).build(conflict_res_table)
        conflict_n = self.max_cnode_num(self.cgraph)
        if conflict_n == 0:
            return longest_latency
        request_ii = int(dfg.synth_params["ii"])
        if request_ii == -1:
            if dfg.ii < conflict_n:
                # TODO: show warnings
                dfg.ii = conflict_n
        elif request_ii < conflict_n:
            fail(
                dfg.region.head.stms[0],
                Errors.RULE_INVALID_II,
                [request_ii, conflict_n],
            )

        for cnode in self.cgraph.get_nodes():
            for dnode in cnode.items:
                self.d2c[dnode] = cnode

        next_candidates = set()
        # sync stms in a cnode
        for cnode in self.cgraph.get_nodes():
            if len(cnode.items) == 1:
                continue
            cnode_begin = max([dnode.begin for dnode in cnode.items])
            for dnode in cnode.items:
                delta = cnode_begin - dnode.begin
                dnode.begin += delta
                dnode.end += delta
                if delta:
                    next_candidates.add(dnode)
        logger.debug("sync dnodes in cnode")
        logger.debug(self.cgraph.nodes)
        cnodes = sorted(
            self.cgraph.get_nodes(),
            key=lambda cn: (cn.items[0].begin, cn.items[0].priority),
        )
        cnode_map = defaultdict(list)
        for cnode in cnodes:
            cnode_map[cnode.res].append(cnode)
        ii = dfg.ii
        while True:
            for res, nodes in cnode_map.items():
                remain_states = [i for i in range(ii)]
                for cnode in nodes:
                    cnode_begin = cnode.items[0].begin
                    state = cnode_begin % ii
                    if state in remain_states:
                        remain_states.remove(state)
                        cnode.state = state
                    else:
                        for i, s in enumerate(remain_states):
                            if state < s:
                                popi = i
                                break
                        else:
                            popi = 0
                        cnode.state = remain_states.pop(popi)
                        offs = ii - (cnode.state + 1)
                        new_begin = (cnode_begin + offs) // ii * ii + cnode.state
                        for dnode in cnode.items:
                            dnode.begin = new_begin
                            next_candidates.add(dnode)
            if next_candidates:
                longest_latency = self._list_schedule_for_pipeline(
                    dfg, next_candidates, longest_latency
                )
                logger.debug(self.cgraph.nodes)
                next_candidates.clear()
            else:
                break
        return longest_latency

    def _node_sched_pipeline(self, dfg, node):
        preds = dfg.preds_without_back(node)
        if preds:
            defuse_preds = dfg.preds_typ_without_back(node, "DefUse")
            usedef_preds = dfg.preds_typ_without_back(node, "UseDef")
            seq_preds = dfg.preds_typ_without_back(node, "Seq")
            sched_times = []
            if seq_preds:
                if node.tag.is_a([JUMP, CJUMP, MCJUMP]) or has_exclusive_function(node.tag):
                    latest_node = max(seq_preds, key=lambda p: p.end)
                    sched_times.append(latest_node.end)
                    logger.debug("latest_node of seq_preds " + str(latest_node))
                else:
                    latest_node = max(seq_preds, key=lambda p: (p.begin, p.end))
                    seq_latency = self.node_seq_latency_map[latest_node]
                    sched_times.append(latest_node.begin + seq_latency)
                    logger.debug("latest_node of seq_preds " + str(latest_node))
                    logger.debug("schedtime " + str(latest_node.begin + seq_latency))
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
        self.regarrays = defaultdict(list)

    def calc_resources(self):
        resources = {}
        for node, ops in self.ops.items():
            for op, num in ops.items():
                if op in resources:
                    resources[op] += num
                else:
                    resources[op] = num
        return resources

    def calc_resources_in_block(self, block):
        resources = {}
        for node, ops in self.ops.items():
            if node.tag.block is not block:
                continue
            for op, num in ops.items():
                if op in resources:
                    resources[op] += num
                else:
                    resources[op] = num
        return resources

    def visit_BINOP(self, ir):
        self.ops[self.current_node][ir.op] += 1
        super().visit_BINOP(ir)

    # def visit_RELOP(self, ir):
    #     self.ops[self.current_node][ir.op] += 1
    #     super().visit_RELOP(ir)

    def visit_CALL(self, ir):
        self.ops[self.current_node][ir.func_scope()] += 1
        func_name = ir.func_scope().name
        if func_name.startswith("polyphony.io.Port") or func_name.startswith("polyphony.io.Queue"):
            inst_ = ir.func.tail()
            self.ports[self.current_node].append(inst_)
        super().visit_CALL(ir)
    
    def visit_PHI(self, ir):
        pass

    def visit_MREF(self, ir):
        if ir.mem.symbol().typ.get_memnode().can_be_reg():
            self.regarrays[self.current_node].append(ir.mem.symbol())
        else:
            self.mems[self.current_node].append(ir.mem.symbol())
        super().visit_MREF(ir)

    def visit_MSTORE(self, ir):
        if ir.mem.symbol().typ.get_memnode().can_be_reg():
            self.regarrays[self.current_node].append(ir.mem.symbol())
        else:
            self.mems[self.current_node].append(ir.mem.symbol())
        super().visit_MSTORE(ir)


class ConflictNode(object):
    READ = 1
    WRITE = 2

    def __init__(self, access, items):
        self.res = ""
        self.access = access
        self.items = items

    @classmethod
    def create(self, dn):
        assert isinstance(dn, DFNode)
        access = ConflictNode.READ if dn.defs else ConflictNode.WRITE
        return ConflictNode(access, [dn])

    @classmethod
    def create_merge_node(self, n0, n1):
        assert isinstance(n0, ConflictNode)
        assert isinstance(n1, ConflictNode)
        return ConflictNode(n0.access | n1.access, n0.items + n1.items)

    @classmethod
    def create_split_node(self, n, items):
        assert isinstance(n, ConflictNode)
        assert set(items) & set(n.items) == set(items)
        for dn in n.items[:]:
            if dn in items:
                n.items.remove(dn)
        n.access = 0
        for dn in n.items:
            n.access |= ConflictNode.READ if dn.defs else ConflictNode.WRITE
        access = 0
        for dn in items:
            access |= ConflictNode.READ if dn.defs else ConflictNode.WRITE
        return ConflictNode(access, list(items))

    def __str__(self):
        access_str = ""
        if self.access & ConflictNode.READ:
            access_str += "R"
        if self.access & ConflictNode.WRITE:
            access_str += "W"
        s = "---- {}, {}, {}\n".format(len(self.items), access_str, self.res)
        s += "\n".join(["  " + str(i) for i in self.items])
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class ConflictGraphBuilder(object):
    def __init__(self, scope, dfg):
        self.scope = scope
        self.dfg = dfg

    def build(self, conflict_res_table):
        cgraphs = self._build_conflict_graphs(conflict_res_table)
        cgraph = self._build_master_conflict_graph(cgraphs)
        return cgraph

    def _build_master_conflict_graph(self, cgraphs):
        master_cgraph = Graph()
        for res, graph in cgraphs.items():
            assert not graph.edges
            accs = []
            for cnode in graph.get_nodes():
                cnode.res = res
                master_cgraph.add_node(cnode)
                accs.append(cnode.access)
            if accs.count(accs[0]) == len(accs) and (
                accs[0] == ConflictNode.READ or accs[0] == ConflictNode.WRITE
            ):
                pass
            else:
                warn(
                    cnode.items[0].tag,
                    Warnings.RULE_PIPELINE_HAS_RW_ACCESS_IN_THE_SAME_RAM,
                    [res],
                )
        # for cnode in master_cgraph.get_nodes():
        #     for dnode in cnode.items:
        #         preds = self.dfg.collect_all_preds(dnode)
        #         if not preds:
        #             continue
        #         for cnode2 in master_cgraph.get_nodes():
        #             if cnode is cnode2:
        #                 continue
        #             if set(preds).intersection(set(cnode2.items)):
        #                 master_cgraph.add_edge(cnode2, cnode)
        logger.debug(master_cgraph.nodes)
        return master_cgraph

    def _build_conflict_graphs(self, conflict_res_table):
        cgraphs = {}
        for res, nodes in conflict_res_table.items():
            if len(nodes) == 1:
                continue
            cgraph = self._build_conflict_graph_per_res(res, nodes)
            cgraphs[res] = cgraph
        return cgraphs

    def _build_conflict_graph_per_res(self, res, conflict_nodes):
        graph = Graph()
        conflict_stms = [n.tag for n in conflict_nodes]
        stm2cnode = {}
        for n in conflict_nodes:
            stm2cnode[n.tag] = ConflictNode.create(n)
        for dn in conflict_nodes:
            graph.add_node(stm2cnode[dn.tag])
        for n0, n1, _ in self.scope.branch_graph.edges:
            if n0 in conflict_stms and n1 in conflict_stms:
                graph.add_edge(stm2cnode[n0], stm2cnode[n1])

        self._merge_same_conditional_branch_nodes(graph, conflict_nodes, stm2cnode)

        logger.debug(graph.nodes)

        def edge_order(e):
            begin0 = max([item.begin for item in e.src.items])
            begin1 = max([item.begin for item in e.dst.items])
            begin = (begin0, begin1) if begin0 <= begin1 else (begin1, begin0)
            distance = (begin1 - begin0) if begin0 <= begin1 else (begin0 - begin1)
            lineno0 = min([item.tag.lineno for item in e.src.items])
            lineno1 = min([item.tag.lineno for item in e.dst.items])
            lineno = (lineno0, lineno1) if lineno0 <= lineno1 else (lineno1, lineno0)
            # In order to avoid crossover edge, 'begin' must be given priority
            return (distance, begin, lineno)

        logger.debug("merging nodes that connected with a branch edge...")
        while graph.edges:
            edges = sorted(graph.edges.orders(), key=edge_order)
            n0, n1, _ = edges[0]
            # logger.debug('merge node')
            # logger.debug(str(n0.items))
            # logger.debug(str(n1.items))
            # logger.debug(str(edges))
            if n0.items[0].begin != n1.items[0].begin and n0.access != n1.access:
                graph.del_edge(n0, n1, auto_del_node=False)
                continue
            cnode = ConflictNode.create_merge_node(n0, n1)
            living_nodes = set()
            for n in graph.nodes:
                adjacents = graph.succs(n).union(graph.preds(n))
                if n0 in adjacents and n1 in adjacents:
                    living_nodes.add(n)
            graph.add_node(cnode)
            graph.del_node(n0)
            graph.del_node(n1)
            for n in living_nodes:
                graph.add_edge(cnode, n)
        logger.debug("after merging")
        logger.debug(graph.nodes)
        return graph

    def _merge_same_conditional_branch_nodes(self, graph, conflict_nodes, stm2cnode):
        conflict_nodes = sorted(conflict_nodes, key=lambda dn: dn.begin)
        for begin, dnodes in itertools.groupby(conflict_nodes, key=lambda dn: dn.begin):
            merge_cnodes = defaultdict(set)
            for dn0, dn1 in itertools.permutations(dnodes, 2):
                stm0 = dn0.tag
                stm1 = dn1.tag
                if stm0 > stm1:
                    continue
                cn0 = stm2cnode[stm0]
                cn1 = stm2cnode[stm1]
                e = graph.find_edge(cn0, cn1)
                if e is not None:
                    continue
                if (stm0.is_a(CMOVE) or stm0.is_a(CEXPR)) and (
                    stm1.is_a(CMOVE) or stm1.is_a(CEXPR)
                ):
                    if stm0.cond == stm1.cond:
                        vs = stm0.cond.find_irs(TEMP)
                        syms = tuple(sorted([v.sym for v in vs]))
                        merge_cnodes[syms].add(cn0)
                        merge_cnodes[syms].add(cn1)
            for cnodes in merge_cnodes.values():
                cnodes = sorted(list(cnodes), key=lambda cn: cn.items[0].tag)
                cn0 = cnodes[0]
                for cn1 in cnodes[1:]:
                    assert cn0.items[0].tag < cn1.items[0].tag
                    mn = ConflictNode.create_merge_node(cn0, cn1)
                    graph.add_node(mn)
                    succs0 = graph.succs(cn0)
                    preds0 = graph.preds(cn0)
                    adjs0 = succs0.union(preds0)
                    succs1 = graph.succs(cn1)
                    preds1 = graph.preds(cn1)
                    adjs1 = succs1.union(preds1)
                    adjs = adjs0.intersection(adjs1)

                    graph.del_node(cn0)
                    graph.del_node(cn1)
                    for adj in adjs:
                        graph.add_edge(mn, adj)
                    cn0 = mn


ADDITION_TIME = 1
MULTIPLICATION_TIME = 3
DIVISION_TIME = 10
class CriticalPathAnalyzer(object):
    def __init__(self, dfg):
        self.dfg = dfg
        self.critical_path = []
        self.critical_path_latency = 0
        self.latency_maps = []
    def analyze(self):
        paths = []
        for path in self.dfg.trace_all_paths(lambda n: self.dfg.succs_typ_without_back(n, "DefUse")):
            latency_map = defaultdict(int)
            paths.append(path)
            for node in path:
                start_time = node.begin
                if node.tag.is_a(MOVE) and node.tag.src.is_a([BINOP, UNOP, RELOP]):
                    op = node.tag.src.op
                    if op in ("Add", "Sub", "BitOr", "BitAnd", "BitXor"):
                        latency = ADDITION_TIME
                    elif op in ("Mult"):
                        latency = MULTIPLICATION_TIME
                    elif op in ("Div", "FloorDiv", "Mod"):
                        latency = DIVISION_TIME
                    elif op in ("And", "Or", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"):
                        latency = ADDITION_TIME
                    else:
                        latency = 0
                else:
                    latency = 0
                if start_time not in latency_map.keys():
                    latency_map[start_time] = latency
                else:
                    latency_map[start_time] += latency
            self.latency_maps.append(latency_map)
        critical_path_index = 0
        critical_path_key = 0
        for i, latency_map in enumerate(self.latency_maps):
            max_latency = max(latency_map.values())
            max_key = max(latency_map, key=latency_map.get)
            if max_latency > self.critical_path_latency:
                critical_path_index = i
                self.critical_path_latency = max_latency
                critical_path_key = max_key
        self.critical_path = [node for node in paths[critical_path_index] if node.begin == critical_path_key]
        split_points = []
        total_latency = 0
        for i, node in enumerate(self.critical_path):
            if node.tag.is_a(MOVE) and node.tag.src.is_a([BINOP, UNOP, RELOP]):
                op = node.tag.src.op
                if op in ("Add", "Sub", "BitOr", "BitAnd", "BitXor"):
                    latency = ADDITION_TIME
                elif op in ("Mult"):
                    latency = MULTIPLICATION_TIME
                elif op in ("Div", "FloorDiv", "Mod"):
                    latency = DIVISION_TIME
                elif op in ("And", "Or", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"):
                    latency = ADDITION_TIME
                else:
                    latency = 0
            else:
                latency = 0
            total_latency += latency
            if total_latency >= MAX_LATENCY:
                split_points.append(i - 1)
                total_latency = latency
        return self.critical_path, self.critical_path_latency, split_points