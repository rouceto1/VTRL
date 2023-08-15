#!/usr/bin/env python3
from process_strategy import process

import os
import random
import numpy as np
import json
import pickle
import itertools
import math
from json import JSONEncoder

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# class strategy containing everything needed to make plan and eventually modify it
class Strategy:
    def __init__(self, uptime, block_size, dataset_weights, place_weights, time_limit, iteration,index ):
        # internal variables: percentage_to_explore, block_size, dataset_weights, place_weights, iteration
        self.uptime = uptime
        self.block_size = block_size
        self.dataset_weights = dataset_weights  # cestlice, strands
        self.place_weights = place_weights  # list of weights for each place TODO: make this strands agnostic
        self.iteration = iteration
        self.time_limit = time_limit
        self.name = self.get_name()
        self.index = index

    def plan_modifier(self, name, experiments_path, old_plan=None, old_strategy=None, uptime=1.0,
                      block_size=1,
                      time_limit=1.0, place_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                      iteration=0,
                      dataset_weight=np.array[0.0, 0.1]):
        total_seasons = [30, 1007]
        total_places = [271, 7]
        places_out_cestlice, c1 = self.make_plan(old_plan, old_strategy, seasons=total_seasons[0],
                                                 places=total_places[0],
                                                 weight=dataset_weight[0],
                                                 uptime=uptime, block_size=block_size,
                                                 time_limit=time_limit,
                                                 place_weights=place_weights, iteration=iteration)

        places_out_strands, c2 = self.make_plan(old_plan, old_strategy, seasons=total_seasons[1],
                                                places=total_places[1],
                                                weight=dataset_weight[1],
                                                uptime=uptime, block_size=block_size,
                                                time_limit=time_limit,
                                                place_weights=place_weights, iteration=iteration)
        # print(percentage_to_explore,c1,c2,c1+c2)
        self.save_plan(name, experiments_path, uptime, block_size, dataset_weight,
                       [places_out_cestlice, places_out_strands],
                       time_limit, places_weights=[], iteration=iteration)
        return [places_out_cestlice, places_out_strands]

    def make_empty_plan(self, seasons, places):
        places_tmp = np.zeros(seasons * places, dtype=int)
        count = 0
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

        last_season = seasons * time_limit
        if last_season > seasons:
            print("Not enough time to make new plan")
            return plan, 0

        available_seasons = last_season - start
        new_season_count = available_seasons * uptime * weight
        rng = np.random.default_rng()
        selected_seasons = rng.choice(np.range(start, last_season), size=int(new_season_count), replace=False)

        for season in range(start, last_season):
            if season in selected_seasons:
                for p in range(places):
                    if self.bool_based_on_probability(place_weights[p]):
                        newly_added += 1
                        plan[season][p] = True

        return plan, newly_added

    def bool_based_on_probability(self, probability=0.5):
        return random.random() < probability

    def save_plan(self, name, experiments_path, uptime, block_size, dataset_weights,
                  plan, time_limit, places_weights=None, iteration=0):
        try:
            os.mkdir(os.path.join(experiments_path, name))
        except FileExistsError:
            # print("Strategy already exists " + os.path.join(experiments_path, name))
            pass
        try:
            os.mkdir(os.path.join(experiments_path, name) + "/plots")
        except FileExistsError:
            pass
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
