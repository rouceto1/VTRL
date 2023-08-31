#!/usr/bin/env python3
import itertools
import os
import random

import numpy as np

from python.teach.planner import Strategy, Mission
from process_strategy import process_new
from json import JSONEncoder
import pickle

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Generator:

    def __init__(self, uptime, block_size, dataset_weights, place_weights, time_limit, time_advance, change_rate,
                 duty_cycle, preteach):
        self.uptime = uptime
        self.block_size = block_size
        self.dataset_weights = dataset_weights  # cestlice, strands
        self.place_weights = place_weights  # list of weights for each place TODO: make this strands agnostic
        self.time_limit = time_limit  # latest possible time to teach
        self.time_advance = time_advance  # how much is each new data training
        self.change_rate = change_rate  # how much to modify TODO make soemthing else then boolean
        self.duty_cycle = duty_cycle
        self.preteach = preteach

        self.mission_count = 0
        self.missions = None


    def make_single_mission(self, strategy):
        self.mission_count += 1
        mission = Mission(self.mission_count)
        mission.c_strategy = strategy
        return mission


    def make_multiple_missions(self):
        a = [self.uptime, self.block_size, self.dataset_weights, self.place_weights, self.time_limit,
           self.time_advance,self.change_rate,self.duty_cycle,self.preteach]
        combinations = list(itertools.product(*a))
        missions = []
        strategies = []
        for combination in combinations:
            strategy = Strategy(uptime=combination[0], block_size=combination[1], dataset_weights=combination[2],
                                place_weights=combination[3], time_limit=combination[4], time_advance=combination[5],
                                change_rate=combination[6], iteration=0, duty_cycle=combination[7],
                                preteach=combination[8])  # TODO make time_advance agnostic of time_limit
            strategies.append(strategy)
        strategies.sort(key=lambda x: x.uptime * x.duty_cycle, reverse=False)

        # this is split to compute supposedly fast strategies first
        for index, strategy in enumerate(strategies):
            mission = Mission(index)
            mission.c_strategy = strategy
            missions.append(mission)
            self.mission_count += 1
        return missions, a

    def save(self,path):
        with open(os.path.join(path, "generator.pkl"), 'wb') as handle:
            pickle.dump(self, handle)

    def print(self):
        #prints out all internal varibles
        print("uptime", self.uptime)
        print("block_size", self.block_size)
        print("dataset_weights", self.dataset_weights)
        print("place_weights", self.place_weights)
        print("time_limit", self.time_limit)
        print("time_advance", self.time_advance)
        print("change_rate", self.change_rate)
        print("duty_cycle", self.duty_cycle)
        print("preteach", self.preteach)
        print("mission_count", self.mission_count)



if __name__ == "__main__":
    uptime_list = np.array([0.25, 0.40, 0.10])

    time_limits = np.array([0.20])
    block_size_list = [1]
    dataset_weights = [np.array([0.0, 1.0])]
    place_weights_contents = [np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2]),
                              np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                              np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
                              ]
    preteach_list = [True, False]
    duty_cycle_list = np.array([1.0, 3.0, 5.0])
    time_advance_list = np.array([0.20])
    change_rate_list = np.array([1.0, 0.0, -1.0])

    gen = Generator(uptime_list, block_size_list, dataset_weights, place_weights_contents, time_limits,
                    time_advance_list, change_rate_list, duty_cycle_list, preteach_list)

    random.seed(42)
    np.random.seed(42)
    gen.missions = gen.make_multiple_missions()

    #random_strategy_2 = Strategy(uptime=0.99, block_size=block_size_list[0], dataset_weights=dataset_weights[0],
    #                             place_weights=np.ones(len(place_weights_contents[0])), time_limit=time_limits[0],
    #                             time_advance=time_advance_list[0],
    #                             change_rate=0.0, iteration=0, duty_cycle=8.0, preteach=True)
    #mission = gen.make_single_mission(random_strategy_2)
    #gen.missions.append(mission)

    process_new(gen, "experiments")
