#!/usr/bin/env python3
import itertools
import os
import random

import numpy as np

from planner import Strategy, Mission
from process_strategy import process_new
from json import JSONEncoder

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def make_multiple_missions():
    time_limits = np.array([0.3])
    block_size_list = [1]
    dataset_weights = [np.array([0.0, 1.0])]
    place_weights_contents = [np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2]),  # outside less
                              np.array([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0]),  # inside less
                              np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0]),
                              np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]),
                              np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                              ]
    uptime_list = np.array([0.30])
    time_advance_list = np.array([0.0])
    change_rate_list = np.array([1.0, 0.0])
    a = [uptime_list, block_size_list, dataset_weights, place_weights_contents, time_limits, time_advance_list,
         change_rate_list]
    combinations = list(itertools.product(*a))
    missions = []
    strategies = []
    for combination in combinations:
        strategy = Strategy(uptime=combination[0], block_size=combination[1], dataset_weights=combination[2],
                            place_weights=combination[3], time_limit=combination[4], time_advance=combination[4],
                            change_rate=combination[6], iteration=0)  # TODO make time_advance agnostic of time_limit
        strategies.append(strategy)
    strategies.sort(key=lambda x: x.uptime * sum(x.place_weights), reverse=False)

    # this is split to compute supposedly fast strategies first
    for index, strategy in enumerate(strategies):
        mission = Mission(index)
        mission.c_strategy = strategy
        missions.append(mission)
    return missions


def make_test_strategy():
    percentage_to_explore = 0.0
    block_size = 1
    whole_place_at_once = False
    single_place_per_batch = False
    place_weight_randomness_list = np.array(np.ones(2))
    a = [[percentage_to_explore, block_size, whole_place_at_once, single_place_per_batch,
          place_weight_randomness_list]]
    return ["none"], a


def make_dummy_strategys(count=10):
    out = []
    names = []
    for i in range(count):
        percentage_to_explore = 0.0
        block_size = 1
        whole_place_at_once = False
        single_place_per_batch = False
        place_weight_randomness_list = np.array(np.ones(2))
        a = [[percentage_to_explore, block_size, whole_place_at_once, single_place_per_batch,
              place_weight_randomness_list]]
        out.append(a[0])
        names.append("0.00_0_0_0_0.00." + str(i))
    return names, out


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    missions = make_multiple_missions()
    process_new(missions, "experiments")
