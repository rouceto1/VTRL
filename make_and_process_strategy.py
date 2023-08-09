#!/usr/bin/env python3
from process_strategy import process

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


def make_places_list_by_places(seasons, percentage_to_explore, block_size, whole_place_at_once,
                               single_place_per_batch,
                               dataset_weight, places, places_weights):
    percentage_to_explore = percentage_to_explore * dataset_weight
    if percentage_to_explore == 0.0:
        # return empty array
        return make_empty_places_list(seasons, places), 0
    else:
        season_count = seasons * percentage_to_explore
        # make numpy array of booleans for each place for each season

        rng = np.random.default_rng()
        selected_seasons = rng.choice(seasons, size=int(season_count), replace=False)
        seasons_out = []
        total_count = 0
        for s in range(seasons):
            if s in selected_seasons:
                tmp = np.zeros(places, dtype=bool)
                for i, p in enumerate(tmp):
                    if bool_based_on_probability(places_weights[i]):
                        tmp[i] = True
                        total_count += 1
                    else:
                        tmp[i] = False
                seasons_out.append(tmp)
            else:
                seasons_out.append(np.zeros(places, dtype=bool))
        return seasons_out, total_count


def bool_based_on_probability(probability=0.5):
    return random.random() < probability


def make_empty_places_list(seasons, places):
    places_tmp = np.zeros(seasons * places, dtype=int)
    count = 0
    seasons_out = []
    for season in range(seasons):
        season_tmp = []
        for place in range(places):
            season_tmp.append(places_tmp[season * places + place])
        seasons_out.append(season_tmp)
    return seasons_out


def make_places_list(seasons, percentage_to_explore, block_size, whole_place_at_once,
                     single_place_per_batch,
                     dataset_weight, places):
    # format of return is: [cestlice[season0[place1 place2 place3 ... place271 ] season1[place1 place2 place3 ... place271 ] ... season30[place1 place2 place3 ... place271 ]]
    #                                       strands[season0[place1 place2 place3 ... place7 ] season1[place1 place2 place3 ... place7 ] ... season1007[place1 place2 place3 ... place7 ]]]

    # make numpy array of booleans for each place for each season
    percentage_to_explore = percentage_to_explore * dataset_weight

    if percentage_to_explore == 0.0:
        # return empty array
        return make_empty_places_list(seasons, places), 0
    else:
        picked_total_count = seasons * percentage_to_explore * places
        block_count = picked_total_count / block_size

        hole_places_count = (seasons * places - picked_total_count)
        hole_size = hole_places_count / block_count
        current_pose = 0
        batch_starts = []
        places_tmp = []
        count = 0
        for i in range(int(block_count)):
            hole = random.randint(0, int(hole_size * 2))
            current_block = current_pose + hole
            if current_block + block_size >= seasons * places:
                # print("generated only " + str(i * block_size) + " out of " + str(picked_total_count) + " :%" + str(
                #    i * block_size / picked_total_count))
                break
            batch_starts.append(current_block)
            current_pose = current_block + block_size

        places_none = np.zeros(places, dtype=int)  # make array that has no places chosen
        possible_places = range(places)
        current_batch_start = 0
        current_batch_size = 0
        for place in range(places * seasons):

            if current_batch_size == block_size:
                current_batch_start = current_batch_start + 1
                current_batch_size = 0

            if current_batch_start >= len(batch_starts):
                places_tmp.append(0)
                continue
            if batch_starts[current_batch_start] > place:
                places_tmp.append(0)
                continue

            current_batch_size = current_batch_size + 1
            count = count + 1
            places_tmp.append(1)

    seasons_out = []
    # split places_tmp into seasons count of arrays
    for season in range(seasons):
        season_tmp = []
        for place in range(places):
            season_tmp.append(places_tmp[season * places + place])
        seasons_out.append(season_tmp)

    return seasons_out, count


def save_strategy(name, experiments_path, percentage_to_explore, block_size, whole_place_at_once,
                  single_place_per_batch, dataset_weights, places_out, places_weights):
    try:
        os.mkdir(os.path.join(experiments_path, name))
    except FileExistsError:
        # print("Strategy already exists " + os.path.join(experiments_path, name))
        pass
    try:
        os.mkdir(os.path.join(experiments_path, name) + "/plots")
    except FileExistsError:
        pass
    with open(os.path.join(experiments_path, name) + "/input.pkl", 'wb') as handle:
        pickle.dump(places_out, handle)
    dictionary = {
        "percents": percentage_to_explore,
        "block_size": block_size,
        "whole_places": whole_place_at_once,
        "uniform_places": single_place_per_batch,
        "dataset_weights": dataset_weights,
        "places_weights": places_weights
    }
    json_object = json.dumps(dictionary, cls=NumpyArrayEncoder)
    with open(os.path.join(experiments_path, name) + "/input.json", "w") as outfile:
        outfile.write(json_object)


def strategy_creator(name, experiments_path, percentage_to_explore, block_size=5,
                     whole_place_at_once=False, single_place_per_batch=False, dataset_weight=np.ones(2)):
    # picked_diff = (1007.0*1007.0-1007.0)/(30.0*30.0-30.0) * 7.0/271.0

    places_out_cestlice, c1 = make_places_list(30, percentage_to_explore, block_size, whole_place_at_once,
                                               single_place_per_batch,
                                               dataset_weight[0], places=271)
    places_out_strands, c2 = make_places_list(1007, percentage_to_explore, block_size, whole_place_at_once,
                                              single_place_per_batch,
                                              dataset_weight[1], places=7)
    # print(percentage_to_explore,c1,c2,c1+c2)
    save_strategy(name, experiments_path, percentage_to_explore, block_size, whole_place_at_once,
                  single_place_per_batch, dataset_weight, [places_out_cestlice, places_out_strands],places_weights=[])
    return [places_out_cestlice, places_out_strands]


def biased_strategy_creator(name, experiments_path, percentage_to_explore, block_size=5,
                            whole_place_at_once=False, single_place_per_batch=False, dataset_weight=np.ones(2),
                            places_weights=None):
    places_out_cestlice, c1 = make_places_list_by_places(30, percentage_to_explore, block_size, whole_place_at_once,
                                                         single_place_per_batch,
                                                         dataset_weight[0], places=271, places_weights=places_weights)
    places_out_strands, c2 = make_places_list_by_places(1007, percentage_to_explore, block_size, whole_place_at_once,
                                                        single_place_per_batch,
                                                        dataset_weight[1], places=8, places_weights=places_weights)
    # print(percentage_to_explore,c1,c2,c1+c2)
    save_strategy(name, experiments_path, percentage_to_explore, block_size, whole_place_at_once,
                  single_place_per_batch, dataset_weight, [places_out_cestlice, places_out_strands], places_weights=places_weights)
    return [places_out_cestlice, places_out_strands]


def make_multiple_strategies(biased=False):
    a = np.arange(0.02, 0.20, 0.02)
    b = np.arange(0.20, 0.50, 0.05)
    c = np.arange(0.50, 0.99, 0.2)
    percentage_to_explore_list = np.concatenate([a, b, c])
    percentage_to_explore_list = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60])
    block_size_list = [1]
    whole_place_at_once_list = [True]
    single_place_per_batch_list = [True]
    dataset_weights = [np.array([0.0, 1.0])]
    place_weights = [np.array([0.0, 1.0])]
    if biased:
        # for strands places: krajnas, office street, office stairs, kitchen, office outside entrance, sofas outside, office outside, office outisde 2
        place_weights_contents = [np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2]),  # outside less
                         np.array([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0]),  # inside less
                         np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0]),  # kitchen less
                         ]
        place_weights = [0, 1, 2]
    a = [percentage_to_explore_list, block_size_list, whole_place_at_once_list, single_place_per_batch_list,
         dataset_weights, place_weights]
    strategies = list(itertools.product(*a))
    names = []
    s = []
    for idx, strategy in enumerate(strategies):
        tmp_strategy = list(strategy)
        # name = f'{strategy[0]:.2f}_{strategy[1]:d}_{strategy[2]:d}_{strategy[3]:d}'
        if not biased:
            name = f'{strategy[0]:.2f}_{strategy[1]:d}_{strategy[2]:d}_{strategy[3]:d}_{strategy[4][0]:.1f}{strategy[4][1]:.1f}'
            s.append(tmp_strategy)
        else:
            name = f'{strategy[0]:.2f}_{strategy[1]:d}_{strategy[2]:d}_{strategy[3]:d}_{strategy[4][0]:.1f}{strategy[4][1]:.1f}_{strategy[5]:d}'
            tmp_strategy[5] = place_weights_contents[strategy[5]]
            s.append(tmp_strategy)
        names.append(name)

    return names, s


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
    names, strategies = make_multiple_strategies(biased=True)
    # names, strategies = make_test_strategy()
    #names, strategies = make_dummy_strategys(10)
    for index, strategy in enumerate(strategies):
        #strategy_creator(names[index], experiments_path, strategy[0], block_size=strategy[1],whole_place_at_once=strategy[2], single_place_per_batch=strategy[3],dataset_weight=strategy[4])
        biased_strategy_creator(names[index], experiments_path, strategy[0], block_size=strategy[1],whole_place_at_once=strategy[2], single_place_per_batch=strategy[3], dataset_weight=strategy[4], places_weights=strategy[5])
    names.sort()
    # names.append("none")
    process(names,"experiments")
