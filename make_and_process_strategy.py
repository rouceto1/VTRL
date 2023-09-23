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


class Mission_generator:

    def __init__(self, uptime, block_size, dataset_weights, place_weights, time_limit, time_advance, change_rate,
                 duty_cycle, preteach, metrics_type,roll_data, ee_ratio,sigma):
        self.uptime = uptime
        self.block_size = block_size
        self.dataset_weights = dataset_weights  # cestlice, strands
        self.place_weights = place_weights  # list of weights for each place TODO: make this strands agnostic
        self.time_limit = time_limit  # latest possible time to teach
        self.time_advance = time_advance  # how much is each new data training
        self.change_rate = change_rate  # how much to modify TODO make soemthing else then boolean
        self.duty_cycle = duty_cycle
        self.preteach = preteach
        self.matrics_type = metrics_type
        self.roll_data = roll_data
        self.ee_ratio = ee_ratio
        self.mission_count = 0
        self.missions = None
        self.sigma = sigma

    def make_single_mission(self, strategy):
        self.mission_count += 1
        mission = Mission(self.mission_count)
        mission.c_strategy = strategy
        return mission

    def make_multiple_missions(self):
        a = [self.uptime, self.block_size, self.dataset_weights, self.place_weights, self.time_limit,
             self.time_advance, self.change_rate, self.duty_cycle, self.preteach, self.matrics_type,self.roll_data, self.ee_ratio,self.sigma]
        combinations = list(itertools.product(*a))
        missions = []
        strategies = []
        for combination in combinations:
            strategy = Strategy(uptime=combination[0], block_size=combination[1], dataset_weights=combination[2],
                                place_weights=combination[3], time_limit=combination[4], time_advance=combination[5],
                                change_rate=combination[6], iteration=0, duty_cycle=combination[7],
                                preteach=combination[8],
                                m_type=combination[9], roll_data=combination[10],
                                ee_ratio=combination[11])
            strategies.append(strategy)
        strategies.sort(key=lambda x: x.uptime * x.duty_cycle, reverse=False)

        # this is split to compute supposedly fast strategies first
        for index, strategy in enumerate(strategies):
            mission = Mission(index)
            mission.c_strategy = strategy
            missions.append(mission)
            self.mission_count += 1
        return missions, a



    def get_txt(self):
        #return string with all internal variables
        txt = ""
        txt += "uptime " + str(self.uptime) + "\n"
        txt += "block_size " + str(self.block_size) + "\n"
        txt += "dataset_weights " + str(self.dataset_weights) + "\n"
        txt += "place_weights " + str(self.place_weights) + "\n"
        txt += "time_limit " + str(self.time_limit) + "\n"
        txt += "time_advance " + str(self.time_advance) + "\n"
        txt += "change_rate " + str(self.change_rate) + "\n"
        txt += "duty_cycle " + str(self.duty_cycle) + "\n"
        txt += "preteach " + str(self.preteach) + "\n"
        txt += "matrics_type " + str(self.matrics_type) + "\n"
        txt += "mission_count " + str(self.mission_count) + "\n"
        txt += "roll_data" + str(self.roll_data) + "\n"
        txt += "ee_ratio" + str(self.ee_ratio) + "\n"
        return txt

    def save_gen(self,path,text):
        with open(os.path.join(path, "generator.txt"), 'w') as handle:
            handle.write(text)

if __name__ == "__main__":
    uptime_list = np.array([0.5])
    #uptime_list = np.array([0.25])
    time_limits = np.array([0.14])
    block_size_list = [1]
    dataset_weights = [np.array([0.0, 1.0])]
    place_weights_contents = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]  # initial weights
    preteach_list = [True, False]
    roll_data_list = [True, False]
    duty_cycle_list = np.array([2.0])
    #duty_cycle_list = np.array([1.0])
    time_advance_list = np.array([0.14])
    change_rate_list = np.array([1.0, 0.0, -1.0])
    #change_rate_list = np.array([1.0])
    metrics_type_list = np.array([0])
    ee_ratio_list = np.array([1.0, 0.5])
    sigma = np.array(range(1))

    gen = Mission_generator(uptime_list, block_size_list, dataset_weights, place_weights_contents, time_limits,
                    time_advance_list, change_rate_list, duty_cycle_list, preteach_list, metrics_type_list, roll_data_list, ee_ratio_list,sigma)

    random.seed(42)
    np.random.seed(42)
    gen.missions, a = gen.make_multiple_missions()

    # random_strategy_2 = Strategy(uptime=0.99, block_size=block_size_list[0], dataset_weights=dataset_weights[0],
    #                             place_weights=np.ones(len(place_weights_contents[0])), time_limit=time_limits[0],
    #                             time_advance=time_advance_list[0],
    #                             change_rate=0.0, iteration=0, duty_cycle=8.0, preteach=True)
    # mission = gen.make_single_mission(random_strategy_2)
    # gen.missions.append(mission)
    process_new(gen, "experiments")
