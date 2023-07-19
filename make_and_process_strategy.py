#!/usr/bin/env python3
from process_strategy import process

import os
import random
import numpy as np
import json
import pickle
import itertools

pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def make_places_list(total_per_place, percentage_to_explore, block_size, whole_place_at_once,
                     single_place_per_batch,
                     place_weight_randomness):
    if percentage_to_explore == 0.0:
        return np.ones(total_per_place, dtype=int)

    picked_places_count = total_per_place * percentage_to_explore
    block_count = picked_places_count / block_size

    hole_places_count = (total_per_place - picked_places_count)
    hole_size = hole_places_count / block_count

    current_pose = 0
    batch_starts = []

    for i in range(int(block_count)):
        hole = random.randint(0, int(hole_size * 2))
        current_block = current_pose + hole
        if current_block + block_size >= total_per_place:
            #print("generated only " + str(i * block_size) + " out of " + str(picked_places_count) + " :%" + str(
            #    i * block_size / picked_places_count))
            break
        batch_starts.append(current_block)
        current_pose = current_block + block_size

    places_out = np.negative(np.ones(total_per_place, dtype=int))  # make array that has no places chosen
    possible_places = range(7)
    for batch in batch_starts:
        rand = random.choices(possible_places, weights=place_weight_randomness)
        for place in range(batch, batch + block_size):
            if whole_place_at_once:
                places_out[place] = -2
                continue
            if not single_place_per_batch:
                rand = random.choices(possible_places, weights=place_weight_randomness)
            places_out[place] = int(rand[0])
    return places_out


def save_strategy(name, experiments_path, percentage_to_explore, block_size, whole_place_at_once,
                  single_place_per_batch, place_weight_randomness, places_out):
    try:
        os.mkdir(os.path.join(experiments_path, name))
    except FileExistsError:
        print("Strategy already exists " + os.path.join(experiments_path, name))
    with open(os.path.join(experiments_path, name) + "/input.pkl", 'wb') as handle:
        pickle.dump(places_out, handle)
    dictionary = {
        "percents": percentage_to_explore,
        "block_size": block_size,
        "whole_places": whole_place_at_once,
        "uniform_places": single_place_per_batch,
        "randomness": place_weight_randomness
    }
    json_object = json.dumps(dictionary, cls=NumpyArrayEncoder)
    with open(os.path.join(experiments_path, name) + "/input.json", "w") as outfile:
        outfile.write(json_object)


def strategy_creator(name, experiments_path, percentage_to_explore, total_per_place=1007, block_size=5,
                     whole_place_at_once=False, single_place_per_batch=False, place_weight_randomness=np.ones(7)):
    places_out = make_places_list(total_per_place, percentage_to_explore, block_size, whole_place_at_once,
                                  single_place_per_batch,
                                  place_weight_randomness)
    save_strategy(name, experiments_path, percentage_to_explore, block_size, whole_place_at_once,
                  single_place_per_batch, place_weight_randomness, places_out)
    return places_out


def make_multiple_strategies():
    a = np.arange(0.02, 0.20, 0.02)
    b = np.arange(0.20, 0.50, 0.05)
    c = np.arange(0.50, 0.99, 0.2)
    percentage_to_explore_list = np.concatenate([a, b, c])
    block_size_list = [1]
    whole_place_at_once_list = [True, False]
    single_place_per_batch_list = [True]
    place_weight_randomness_list = np.array(np.ones(7))
    a = [percentage_to_explore_list, block_size_list, whole_place_at_once_list, single_place_per_batch_list,
         [place_weight_randomness_list]]
    strategies = list(itertools.product(*a))
    names = []
    for strategy in strategies:
        #name = f'{strategy[0]:.2f}_{strategy[1]:d}_{strategy[2]:d}_{strategy[3]:d}'
        name = f'{strategy[0]:.2f}_{strategy[2]:d}'
        names.append(name)
    return names, strategies

def make_test_strategy():
    percentage_to_explore = 0.1
    block_size = 1
    whole_place_at_once = False
    single_place_per_batch = False
    place_weight_randomness_list = np.array(np.ones(7))
    a = [[percentage_to_explore, block_size, whole_place_at_once, single_place_per_batch,
         place_weight_randomness_list]]
    return ["test"], a



if __name__ == "__main__":
    names, strategies = make_multiple_strategies()
    #names, strategies = make_test_strategy()
    for index, strategy in enumerate(strategies):
        strategy_creator(names[index], experiments_path, strategy[0], block_size=strategy[1],
                         whole_place_at_once=strategy[2], single_place_per_batch=strategy[3],
                         place_weight_randomness=strategy[4])
    process(names)
