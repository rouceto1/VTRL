#!/usr/bin/env python3

import json
import os
import pickle
import random
from json import JSONEncoder
import copy
import numpy as np

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "planner":
            renamed_module = "python.teach.planner"
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


class Mission:
    def __init__(self, index, description=None):
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
        #print("Loaded mission: " + str(os.path.join(mission_folder, "mission.pickle")))
        return m

    def setup_current_strategy(self, init_weights=None):
        self.c_strategy.setup_strategy(self.mission_folder, init_weights)

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
        return False

    def advance_mission(self, metrics):
        self.add_new_strategy(copy.deepcopy(self.c_strategy))
        self.c_strategy.strategy_modifier(metrics)
        self.c_strategy.progress = 0
        self.setup_current_strategy()
        self.plan_modifier(self.old_strategies[-1].plan, self.old_strategies[-1])

    def plan_modifier(self, old_plan=[None, None], old_strategy=None):
        total_seasons = [30, 1007]
        total_places = [271, 8]
        places_out_cestlice, c1 = self.make_plan(old_plan[0], old_strategy, seasons=total_seasons[0],
                                                 places=total_places[0],
                                                 weight=self.c_strategy.dataset_weights[0],
                                                 uptime=self.c_strategy.uptime, block_size=self.c_strategy.block_size,
                                                 time_limit=self.c_strategy.time_limit,
                                                 place_weights=self.c_strategy.place_weights,
                                                 exploitation_weights=self.c_strategy.process_weights(
                                                     duty_cycle=self.c_strategy.duty_cycle),
                                                 iteration=self.c_strategy.iteration,
                                                 rolling=self.c_strategy.roll_data)

        places_out_strands, c2 = self.make_plan(old_plan[1], old_strategy, seasons=total_seasons[1],
                                                places=total_places[1],
                                                weight=self.c_strategy.dataset_weights[1],
                                                uptime=self.c_strategy.uptime, block_size=self.c_strategy.block_size,
                                                time_limit=self.c_strategy.time_limit,
                                                place_weights=self.c_strategy.place_weights,
                                                exploitation_weights=self.c_strategy.process_weights(
                                                    duty_cycle=self.c_strategy.duty_cycle),
                                                iteration=self.c_strategy.iteration,
                                                rolling=self.c_strategy.roll_data)
        # print(percentage_to_explore,c1,c2,c1+c2)
        self.save_plan(self.name, self.experiments_path, self.c_strategy)
        self.c_strategy.plan = [places_out_cestlice, places_out_strands]
        if c1 + c2 == 0:
            self.no_more_data = True
        return [places_out_cestlice, places_out_strands], [c1, c2]

    def make_empty_plan(self, seasons, places):
        places_tmp = np.zeros(seasons * places, dtype=int)
        seasons_out = []
        for season in range(seasons):
            season_tmp = []
            for place in range(places):
                season_tmp.append(places_tmp[season * places + place])
            seasons_out.append(season_tmp)
        return seasons_out

    def make_plan(self, old_plan, old_strategy, seasons, places, weight, uptime=1.0,
                  place_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                  exploitation_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                  block_size=1, time_limit=1.0, iteration=0, rolling=False, ee_ratio=1.0):
        plan = self.make_empty_plan(seasons, places)
        if uptime * weight == 0.0:
            return plan, 0
        start = 0
        if old_plan is not None:
            start = int(old_strategy.time_limit * seasons)
            if not rolling:
                plan = old_plan

        last_season = int(seasons * time_limit)
        if last_season > seasons:
            print("Not enough time to make new full plan")
            last_season = seasons
            return plan, 0

        available_seasons = last_season - start
        new_season_count = available_seasons * uptime * weight
        rng = np.random.default_rng()
        np.random.seed(42)

        selected_seasons = rng.choice(range(start, last_season), size=int(new_season_count), replace=False)
        newly_added = 0
        r = np.random.rand(len(plan), len(plan[0]))
        for season in range(start, last_season):
            if season in selected_seasons:
                random_ee = random.random() > ee_ratio #if exploring only ee_ratio is 1, therefore this is always false
                for p in range(places):
                    if random_ee:
                        if r[season][p] < exploitation_weights[p]:
                            newly_added += 1
                            plan[season][p] = True
                    else:
                        if r[season][p] < place_weights[p]:
                            newly_added += 1
                            plan[season][p] = True

        return plan, newly_added

    def save_plan(self, name, experiments_path, strategy):
        iteration = "_" + str(strategy.iteration)
        with open(os.path.join(experiments_path, name) + "/plan" + iteration + ".pkl", 'wb') as handle:
            pickle.dump(strategy.plan, handle)
        dictionary = {
            "uptime": strategy.uptime,
            "block_size": strategy.block_size,
            "dataset_weights": strategy.dataset_weights,
            "place_weights": strategy.place_weights,
            "time_limit": strategy.time_limit,
            "iteration": strategy.iteration,
            "time_advance": strategy.time_advance,
            "change_rate": strategy.change_rate,
            "duty_cycle": strategy.duty_cycle,
            "preteach": strategy.preteach,
            "roll_data": strategy.roll_data,
            "metrics_type": int(strategy.metrics_type),
            "ee_ratio": str(strategy.ee_ratio)
        }
        json_object = json.dumps(dictionary, cls=NumpyArrayEncoder)
        with open(os.path.join(experiments_path, name) + "/strategy" + iteration + ".json", "w") as outfile:
            outfile.write(json_object)


# class strategy containing everything needed to make plan and eventually modify it
class Strategy:
    def __init__(self, uptime=None, block_size=None, dataset_weights=None, place_weights=None, time_limit=None,
                 time_advance=None, change_rate=None,
                 iteration=None, duty_cycle=None, preteach=None, m_type=None, roll_data=None, ee_ratio=None):
        # internal variables: percentage_to_explore, block_size, dataset_weights, place_weights, iteration
        self.uptime = uptime
        self.block_size = block_size
        self.dataset_weights = dataset_weights  # cestlice, strands
        self.place_weights = place_weights  # list of weights for each place TODO: make this strands agnostic
        self.time_limit = time_limit  # latest possible time to teach
        self.iteration = iteration
        self.time_advance = time_advance  # how much is each new data training
        self.change_rate = change_rate  # how much to modify TODO make soemthing else then boolean
        self.duty_cycle = duty_cycle
        self.preteach = preteach
        self.roll_data = roll_data
        self.ee_ratio = ee_ratio
        self.metrics_type = m_type
        # 0: entropies[i] < mean + std * 0.1:
        #    well_understood.append(file_list[i])
        # boolean
        # 1:


        if place_weights is not None:
            self.place_weights = self.process_weights(self.place_weights, np.ones(8), self.duty_cycle)

        self.plan = None
        self.used_teach_count = 0
        self.file_list = None
        self.model_path = None
        self.estimates_path = None
        self.grading_path = None
        self.usage_path = None
        self.metrics_path = None
        self.train_time = None
        self.eval_time = None
        self.grading = [None, None]
        self.is_faulty = False
        self.progress = 0

        self.next_metrics = None

    def get_given_teach_count(self):
        return len(self.file_list)

    def print_parameters(self):
        print(
            f"Up: {self.uptime}, Bs: {self.block_size}, pw: {np.array2string(self.place_weights, precision=2, floatmode='fixed')}, lim e: {self.time_limit}, "
            f"it: {self.iteration}, cr: {self.change_rate}, ta {self.time_advance}, pt {self.preteach}")

    def title_parameters(self):
        if self.place_weights is None:
            return f"U:{self.uptime},i:{self.iteration},c:{self.change_rate}"
        return f"U:{self.uptime},{np.array2string(self.place_weights, precision=1, floatmode='fixed')},i:{self.iteration}:c:{int(self.change_rate)}"

    def setup_strategy(self, path, init_weights):
        self.model_path = os.path.join(path, str(self.iteration) + "_weights.pt")
        self.estimates_path = os.path.join(path, str(self.iteration) + "_estimates.pkl")
        self.grading_path = os.path.join(path, str(self.iteration) + "_grading.pkl")
        self.usage_path = os.path.join(path, str(self.iteration) + "_usage.png")
        self.metrics_path = os.path.join(path, str(self.iteration) + "_metrics.png")

    def strategy_modifier(self, metrics):
        # gives back new place weight multiplier besed on metrics
        # keeping the same exploration ratio
        self.time_limit += self.time_advance
        self.iteration += 1

        if self.change_rate == 0.0:
            return
        if self.change_rate == -1.0:
            np.random.seed()
            self.place_weights = self.process_weights(np.random.rand(len(self.place_weights)),
                                                      np.random.rand(len(self.place_weights)),
                                                      self.duty_cycle)
            np.random.seed(42)
            return
        if self.iteration == 1:
            self.place_weights = self.process_weights(np.ones(8), metrics, self.duty_cycle)
        else:
            self.place_weights = self.process_weights(self.place_weights, metrics, self.duty_cycle)

    def process_weights(self, weights=np.ones(8), metrics=np.ones(8), duty_cycle=1.0):
        ratio = sum(weights * metrics)
        new_weights = weights * metrics * (duty_cycle / ratio)
        return self.clip_weights(new_weights)

    def clip_weights(self, weights):
        # clip weights to be between 0 and 1, distribute teh rest between
        remainder = 0.0
        # clip any weights to 1 and distribute teh reminder to the other weights
        for i in range(len(weights)):
            if weights[i] > 1.0:
                remainder += weights[i] - 1.0
                weights[i] = 1.0
        # distribute the reminder evenly between the other non one weights
        if remainder == 0.0:
            return weights
        none_one = sum(weights < 1.0)
        for i in range(len(weights)):
            if weights[i] < 1.0:
                weights[i] += remainder / none_one

        return self.clip_weights(weights)
