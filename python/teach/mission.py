#!/usr/bin/env python3

import json
import os
import pickle
import random
from json import JSONEncoder
import copy
import numpy as np
from python.teach.planner import VTRL, VTRE

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "mission":
            renamed_module = "python.teach.mission"
        if name == "Mission" or name == "Strategy":
                renamed_module = "python.teach.mission"
        if module == "python.grade_results":
            renamed_module = "python.grading.grade_results"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def save_plan(name, experiments_path, strategy):
    iteration = "_" + str(strategy.iteration)
    with open(os.path.join(experiments_path, name) + "/timetable" + iteration + ".pkl", 'wb') as handle:
        pickle.dump(strategy.timetable, handle)
    dictionary = {
        "uptime": strategy.uptime,
        "block_size": strategy.block_size,
        "dataset_weights": strategy.dataset_weights,
        "preferences": strategy.preferences,
        "time_limit": strategy.time_limit,
        "iteration": strategy.iteration,
        "time_advance": strategy.time_advance,
        "change_rate": strategy.change_rate,
        "duty_cycle": strategy.duty_cycle,
        "preteach": strategy.preteach,
        "roll_data": strategy.roll_data,
        "method_type": int(strategy.method_type),
        "ee_ratio": str(strategy.ee_ratio)
    }
    json_object = json.dumps(dictionary, cls=NumpyArrayEncoder)
    with open(os.path.join(experiments_path, name) + "/strategy" + iteration + ".json", "w") as outfile:
        outfile.write(json_object)


class Mission:
    def __init__(self, index, description=None, simulation=False):
        self.index = index
        self.name = f"{index:02d}"
        self.c_strategy = None
        self.old_strategies = []
        self.experiments_path = None
        self.plot_folder = None
        self.mission_folder = None
        self.description = description
        self.set_up = False
        self.no_more_data = False
        self.simulation = simulation

    def save(self):
        # save mission to pickle? maybe mocap?
        self.c_strategy.file_list = None
        for strategy in self.old_strategies:
            strategy.file_list = None
        with open(os.path.join(self.mission_folder, "mission.pickle"), 'wb') as f:
            pickle.dump(self, f)
        print("saved mission: " + str(os.path.join(self.mission_folder, "mission.pickle")))

    def load(self, mission_folder):
        with open(os.path.join(mission_folder, "mission.pickle"), 'rb') as f:
            m = renamed_load(f)
        # print("Loaded mission: " + str(os.path.join(mission_folder, "mission.pickle")))
        return m

    def setup_current_strategy(self):
        self.c_strategy.setup_strategy(self.mission_folder)

    def add_new_strategy(self, strategy):
        self.old_strategies.append(self.c_strategy)
        self.c_strategy = strategy

    def setup_mission(self, exp_folder_path):
        self.experiments_path = exp_folder_path
        self.mission_folder = os.path.join(self.experiments_path, self.name)
        try:
            os.mkdir(self.mission_folder)
        except FileExistsError:
            print("Mission already exists " + os.path.join(experiments_path))
            return False
        try:
            self.plot_folder = os.path.join(self.experiments_path, self.name) + "/plots"
            os.mkdir(self.plot_folder)
        except FileExistsError:
            return False
        self.set_up = True
        self.setup_current_strategy()  # sets up current mission
        self.no_more_data = self.c_strategy.advance(None, None, None)

        save_plan(self.name, self.experiments_path, self.c_strategy)
        self.c_strategy.print_parameters()
        self.save()
        return True

    def advance_mission(self, ambiguity):
        self.add_new_strategy(copy.deepcopy(self.c_strategy))
        self.c_strategy.iteration += 1
        self.setup_current_strategy()
        self.c_strategy.progress = 0
        self.no_more_data = self.c_strategy.advance(ambiguity, self.old_strategies[-1].timetable, self.old_strategies[-1])
        save_plan(self.name, self.experiments_path, self.c_strategy)


# class strategy containing everything needed to make timetable and eventually modify it
class Strategy:
    def __init__(self, uptime=None, block_size=None, dataset_weights=None, preferences=None, time_limit=None,
                 time_advance=None, change_rate=None,
                 iteration=None, duty_cycle=None, preteach=None, m_type=None, roll_data=None, ee_ratio=None,
                 simulation=False):
        # internal variables: percentage_to_explore, block_size, dataset_weights, preferences, iteration
        self.no_more_data = None
        self.exploit = None
        self.uptime = uptime
        self.block_size = block_size
        self.dataset_weights = dataset_weights  # cestlice, strands
        self.preferences = preferences  # list of weights for each place TODO: make this strands agnostic
        self.time_limit = time_limit  # latest possible time to teach
        self.time_start = 0
        self.iteration = iteration
        self.time_advance = time_advance  # how much is each new data training
        self.change_rate = change_rate  # how much to modify TODO make soemthing else then boolean
        self.duty_cycle = duty_cycle
        self.preteach = preteach
        self.roll_data = roll_data
        self.ee_ratio = ee_ratio
        self.method_type = m_type

        if not simulation:
            self.planner = VTRL()
            if preferences is not None:
                self.preferences = self.planner.advance_preferences(self.preferences, [np.ones(271),np.ones(8)], self.duty_cycle)
        else:
            self.planner = VTRE()

        self.timetable = None
        self.used_teach_count = 0
        self.file_list = None
        self.model_path = None
        self.estimates_path = None
        self.grading_path = None
        self.usage_path = None
        self.ambiguity_path = None
        self.train_time = None
        self.eval_time = None
        self.grading = [None, None, None]
        self.is_faulty = False
        self.progress = 0

        self.next_ambiguity = None

        self.map_count = 2

    def get_given_teach_count(self):
        return len(self.file_list)

    def print_parameters(self):
        print(
            f"Up: {self.uptime}, Bs: {self.block_size}, lim e: {self.time_limit}, "
            f"it: {self.iteration}, cr: {self.change_rate}, ta {self.time_advance}, pt {self.preteach}, rol{self.roll_data}, ee {self.ee_ratio}, m_type {self.method_type}")

    def title_parameters(self):
        if self.preferences is None:
            return f"U:{self.uptime},i:{self.iteration},c:{self.change_rate}"
        return f"U:{self.uptime},i:{self.iteration}:c:{int(self.change_rate)}"

    def setup_strategy(self, path):
        self.model_path = os.path.join(path, str(self.iteration) + "_weights.pt")
        self.estimates_path = os.path.join(path, str(self.iteration) + "_estimates.pkl")
        self.grading_path = os.path.join(path, str(self.iteration) + "_grading.pkl")
        self.usage_path = [os.path.join(path, str(self.iteration) + "0_usage.png"),os.path.join(path, str(self.iteration) + "1_usage.png")]

        self.ambiguity_path = [os.path.join(path, str(self.iteration) + "_0_ambiguity.png"),os.path.join(path, str(self.iteration) + "_1_ambiguity.png")]

    def advance(self, ambiguity, old_timetable, old_strategy):

        if ambiguity is not None:
            self.time_start = self.time_limit
            self.time_limit += self.time_advance
            self.preferences = self.planner.get_preferences_for_next_round(ambiguity, self)

        if old_timetable is None:
            self.timetable, _, self.exploit, self.no_more_data = self.planner.timetable_modifier_vtrl(self)
        else:
            self.timetable, _, self.exploit, self.no_more_data = self.planner.timetable_modifier_vtrl(self,
                                                                                                      old_timetable=old_timetable,
                                                                                                      old_strategy=old_strategy)
        return self.no_more_data
