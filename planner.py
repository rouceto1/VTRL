#!/usr/bin/env python3

import json
import os
import pickle
import random
from json import JSONEncoder

import numpy as np

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Mission:
    def __init__(self, index, description=None):
        self.index = index
        self.name = str(index)
        self.c_strategy = None
        self.old_strategies = []
        self.experiments_path = None
        self.plot_folder = None
        self.mission_folder = None
        self.description = description
        self.set_up = False
    def save(self):
        #save mission to pickle? maybe mocap?
        self.c_strategy.file_list = None
        with open(os.path.join(self.mission_folder,"mission.pickle"), 'wb') as f:
            pickle.dump(self, f)

    def load(self, mission_folder):
        with open(os.path.join(mission_folder,"mission.pickle"), 'rb') as f:
            m = pickle.load(f)
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
            # print("Strategy already exists " + os.path.join(experiments_path, name))
            pass
        try:
            self.plot_folder = os.path.join(self.experiments_path, self.name) + "/plots"
            os.mkdir(self.plot_folder)
        except FileExistsError:
            pass
        self.set_up = True


    def plan_modifier(self, old_plan=None, old_strategy=None):
        total_seasons = [30, 1007]
        total_places = [271, 7]
        places_out_cestlice, c1 = self.make_plan(old_plan, old_strategy, seasons=total_seasons[0],
                                                 places=total_places[0],
                                                 weight=self.c_strategy.dataset_weights[0],
                                                 uptime=self.c_strategy.uptime, block_size=self.c_strategy.block_size,
                                                 time_limit=self.c_strategy.time_limit,
                                                 place_weights=self.c_strategy.place_weights,
                                                 iteration=self.c_strategy.iteration)

        places_out_strands, c2 = self.make_plan(old_plan, old_strategy, seasons=total_seasons[1],
                                                places=total_places[1],
                                                weight=self.c_strategy.dataset_weights[1],
                                                uptime=self.c_strategy.uptime, block_size=self.c_strategy.block_size,
                                                time_limit=self.c_strategy.time_limit,
                                                place_weights=self.c_strategy.place_weights,
                                                iteration=self.c_strategy.iteration)
        # print(percentage_to_explore,c1,c2,c1+c2)
        self.save_plan(self.name, self.experiments_path, self.c_strategy.uptime, self.c_strategy.block_size,
                       self.c_strategy.dataset_weights,
                       [places_out_cestlice, places_out_strands],
                       self.c_strategy.time_limit, places_weights=self.c_strategy.place_weights,
                       iteration=self.c_strategy.iteration)
        self.c_strategy.plan = [places_out_cestlice, places_out_strands]
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
                  block_size=1, time_limit=1.0, iteration=0):
        plan = self.make_empty_plan(seasons, places)
        if uptime * weight == 0.0:
            return plan, 0
        start = 0
        newly_added = 0
        if old_plan is not None:
            # get index of  first column of old_plan that is all zeros TODO get this from old_strategy
            for i in range(len(old_plan[0])):
                if not np.any(old_plan[:, i]):
                    start = i
                    break
            plan = old_plan

        last_season = int(seasons * time_limit)
        if last_season > seasons:
            print("Not enough time to make new plan")
            return plan, 0

        available_seasons = last_season - start
        new_season_count = available_seasons * uptime * weight
        rng = np.random.default_rng()
        np.random.seed(42)

        selected_seasons = rng.choice(range(start, last_season), size=int(new_season_count), replace=False)

        for season in range(start, last_season):
            if season in selected_seasons:
                for p in range(places):
                    if self.bool_based_on_probability(place_weights[p]):
                        newly_added += 1
                        plan[season][p] = True

        return plan, newly_added

    def bool_based_on_probability(self, probability=0.5):
        random.seed(42)

        return random.random() < probability

    def save_plan(self, name, experiments_path, uptime, block_size, dataset_weights,
                  plan, time_limit, places_weights=None, iteration=0):
        iteration = "_" + str(iteration)
        with open(os.path.join(experiments_path, name) + "/plan" + iteration + ".pkl", 'wb') as handle:
            pickle.dump(plan, handle)
        dictionary = {
            "uptime_percent": uptime,
            "block_size": block_size,
            "dataset_weights": dataset_weights,
            "places_weights": places_weights,
            "time_limit": time_limit
        }
        json_object = json.dumps(dictionary, cls=NumpyArrayEncoder)
        with open(os.path.join(experiments_path, name) + "/strategy" + iteration + ".json", "w") as outfile:
            outfile.write(json_object)


# class strategy containing everything needed to make plan and eventually modify it
class Strategy:
    def __init__(self, uptime, block_size, dataset_weights, place_weights, time_limit, iteration):
        # internal variables: percentage_to_explore, block_size, dataset_weights, place_weights, iteration
        self.uptime = uptime
        self.block_size = block_size
        self.dataset_weights = dataset_weights  # cestlice, strands
        self.place_weights = place_weights  # list of weights for each place TODO: make this strands agnostic
        self.time_limit = time_limit
        self.iteration = iteration
        self.plan = None
        self.used_teach_count = 0
        self.file_list = None
        self.model_path = None
        self.estimates_path = None
        self.grading_path = None
        self.training_time = None
        self.eval_time = None
        self.grading = []
        self.is_faulty = False


    def get_given_teach_count(self):
        return len(self.file_list)

    def print_parameters(self):
        print(f"Uptime: {self.uptime}, Block size: {self.block_size}, place_weights: {self.place_weights}, time_limit: {self.time_limit}, iteration: {self.iteration}")
    def setup_strategy(self, path):
        self.model_path = os.path.join(path, str(self.iteration) + "_weights.pt")
        self.estimates_path = os.path.join(path, str(self.iteration) + "_estimates.pkl")
        self.grading_path = os.path.join(path, str(self.iteration) + "_grading.pkl")
