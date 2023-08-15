#!/usr/bin/env python3
from process_strategy import process
from planner import Strategy

import os
import random
import numpy as np
import json
import pickle
import itertools
import math

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def make_multiple_strategies(biased=False):
    a = np.arange(0.02, 0.20, 0.02)
    b = np.arange(0.20, 0.50, 0.05)
    c = np.arange(0.50, 0.99, 0.2)
    time_limits = np.array( [0.4,0.5,0.6,0.7,0.8,0.9])
    block_size_list = [1]
    whole_place_at_once_list = [True]
    single_place_per_batch_list = [True]
    dataset_weights = [np.array([0.0, 1.0])]
    place_weights = [np.array([0.0, 1.0])]
    place_weights_contents = [np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2]),  # outside less
                                  np.array([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0]),  # inside less
                                  np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0]),  # kitchen less
                                  ]
    place_weights_contents = [np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                  np.array([0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                  np.array([0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                  np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]),
                                  np.array([0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]),
                                  np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1]),
                                  np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1]),
                                  np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]),
                                  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                                  ]
    uptime_list = np.array([0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
    a = [uptime_list, block_size_list,dataset_weights, place_weights_contents,time_limits]
    combinations = list(itertools.product(*a))
    strategies=[]
    for index, combination in enumerate(combinations):
        strategies.append(Strategy(combination[0], combination[1], combination[2], combination[3], combination[4],iteration=0,index))
    return strategies


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
    strategies = make_multiple_strategies(biased=True)
    strategies.sort(key=lambda x: x.index, reverse=True)
    process(names, "experiments")
