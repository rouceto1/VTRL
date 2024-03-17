#!/usr/bin/env python3
from abc import ABC, abstractmethod

import json
import os
import pickle
import random
from json import JSONEncoder
import copy
import numpy as np
from copy import deepcopy


class Planner(ABC):
    @abstractmethod
    def timetable_modifier_vtrl(self, strategy, old_timetable=None, old_strategy=None):
        """
        Make empty version of timetable
        """

        raise NotImplementedError

    @abstractmethod
    def get_preferences_for_next_round(self, ambiguity, strategy):
        """
        Create new preferences
        """
        raise NotImplementedError


class VTRE(Planner):
    def timetable_modifier_vtrl(self, strategy, old_timetable=None, old_strategy=None):
        if old_timetable is None:
            # This is initial search for timetable
            return self.random_search(strategy.time_start, strategy.time_limit, strategy.duty_cycle)
        else:
            # if strategy should be random
            if strategy.change_rate == -1.0:
                return self.random_search(strategy.time_start, strategy.time_limit, strategy.duty_cycle, old_timetable)
            # This should generate timetable for next round
            # TODO: implement this
            return self.random_search(strategy.time_start, strategy.time_limit, strategy.duty_cycle, old_timetable)

    def get_preferences_for_next_round(self, ambiguity, strategy):
        # gives back new place preferences multiplier besed on ambiguity
        # keeping the same exploration ratio
        if strategy.change_rate == 0.0:
            return strategy.preferences

        # TODO implement
        if strategy.iteration == 1:
            return strategy.preferences
        else:
            return strategy.preferences
            # return self.advance_preferences(strategy.preferences, ambiguity, strategy.duty_cycle)

    def random_search(self, start, stop, duty_cycle, old_timetable=None):
        # make first timetable randomly
        if old_timetable is not None:
            timetable = deepcopy(old_timetable)
        else:
            timetable = []
        counts = [0, 0]
        exploits = []
        total = 0
        false = [False, False]
        maps = [[True, False], [False, True]]
        last = 0
        for i in range(30):
            if last == random.random():
                print("BROKEN RANDOM NUMBERS IF THIS OCCURS MULTIPLE TIMES")
            last = random.random()
        for i in range(start, stop):
            if random.random() < duty_cycle:
                current = random.randint(0, 1)
                timetable.append(maps[current])
                total += 1
                counts[current] += 1
                exploits.append(maps[current])
            else:
                timetable.append(false)
                exploits.append(false)
        return timetable, counts, exploits, total


class VTRL(Planner):

    def timetable_modifier_vtrl(self, strategy, old_timetable=None, old_strategy=None):
        if old_timetable is None:
            old_timetable = [None, None]
        total_seasons = [30, 1007]
        total_places = [271, 8]
        exploit_weights = self.advance_preferences([np.ones(271), np.ones(8)], [np.ones(271), np.ones(8)],
                                                   duty_cycle=strategy.duty_cycle)
        places_out_cestlice, exploit_cestlice, c1 = self.make_timetable_vtrl(old_timetable[0], old_strategy,
                                                                             seasons=total_seasons[0],
                                                                             places=total_places[0],
                                                                             weight=strategy.dataset_weights[0],
                                                                             uptime=strategy.uptime,
                                                                             block_size=strategy.block_size,
                                                                             time_limit=strategy.time_limit,
                                                                             place_weights=strategy.preferences[0],
                                                                             exploitation_weights=exploit_weights[0],
                                                                             iteration=strategy.iteration,
                                                                             rolling=strategy.roll_data,
                                                                             ee_ratio=strategy.ee_ratio)

        places_out_strands, exploit_strands, c2 = self.make_timetable_vtrl(old_timetable[1], old_strategy,
                                                                           seasons=total_seasons[1],
                                                                           places=total_places[1],
                                                                           weight=strategy.dataset_weights[1],
                                                                           uptime=strategy.uptime,
                                                                           block_size=strategy.block_size,
                                                                           time_limit=strategy.time_limit,
                                                                           place_weights=strategy.preferences[1],
                                                                           exploitation_weights=exploit_weights[1],
                                                                           iteration=strategy.iteration,
                                                                           rolling=strategy.roll_data,
                                                                           ee_ratio=strategy.ee_ratio)
        print(c1, c2, c1 + c2)
        return [places_out_cestlice, places_out_strands], [c1, c2], [exploit_cestlice, exploit_strands], c1 + c2 == 0

    def make_timetable_vtrl(self, old_timetable, old_strategy, seasons, places, weight, uptime=1.0,
                            place_weights=None,
                            exploitation_weights=None,
                            block_size=1, time_limit=1.0, iteration=0, rolling=False, ee_ratio=1.0):
        timetable = self.make_empty_timetable(seasons, places)
        exploit_timetable = self.make_empty_timetable(seasons, places)
        if uptime * weight == 0.0:
            return timetable, exploit_timetable, 0
        start = 0
        if old_timetable is not None:
            start = int(old_strategy.time_limit * seasons)
            if not rolling:
                timetable = old_timetable

        last_season = int(seasons * time_limit)
        if last_season > seasons:
            print("Not enough time to make new full timetable")
            last_season = seasons
            return timetable, exploit_timetable, 0

        available_seasons = last_season - start
        new_season_count = available_seasons * uptime * weight
        rng = np.random.default_rng()
        np.random.seed(42)

        selected_seasons = rng.choice(range(start, last_season), size=int(new_season_count), replace=False)
        newly_added = 0
        r = np.random.rand(len(timetable), len(timetable[0]))
        for season in range(start, last_season):
            if season in selected_seasons:
                random_ee = random.random() > ee_ratio  # if exploring only ee_ratio is 1, therefore this is always false
                for p in range(places):
                    if random_ee:
                        if r[season][p] < exploitation_weights[p]:
                            newly_added += 1
                            timetable[season][p] = True
                            exploit_timetable[season][p] = True
                    else:
                        if r[season][p] < place_weights[p]:
                            newly_added += 1
                            timetable[season][p] = True

        return timetable, exploit_timetable, newly_added

    def make_empty_timetable(self, seasons, places):
        places_tmp = np.zeros(seasons * places, dtype=int)
        seasons_out = []
        for season in range(seasons):
            season_tmp = []
            for place in range(places):
                season_tmp.append(places_tmp[season * places + place])
            seasons_out.append(season_tmp)
        return seasons_out

    def get_preferences_for_next_round(self, ambiguity, strategy):
        # gives back new place preferences multiplier besed on ambiguity
        # keeping the same exploration ratio

        if strategy.change_rate == 0.0:
            return strategy.preferences
        if strategy.change_rate == -1.0:
            np.random.seed()
            pref = self.advance_preferences(
                [np.random.rand(len(strategy.preferences[0])), np.random.rand(len(strategy.preferences[1]))],
                [np.random.rand(len(strategy.preferences[0])), np.random.rand(len(strategy.preferences[1]))],
                strategy.duty_cycle)
            np.random.seed(42)
            return pref
        # if strategy.iteration == 1:
        # return self.advance_preferences([np.ones(271), np.ones(8)], ambiguity, strategy.duty_cycle)
        # else:
        return self.advance_preferences(strategy.preferences, ambiguity, strategy.duty_cycle)

    def advance_preferences(self, preferences=None, metrics=None, duty_cycle=1.0):
        pref = []
        for idx, p in enumerate(preferences):
            obtained_preferences = p * metrics[idx]
            ratio = sum(obtained_preferences)
            if ratio == 0.0:
                print("RATIO ZERO")
                return preferences
            new_preferences = obtained_preferences * (duty_cycle * len(metrics[idx]) / ratio)
            pref.append(clip_preferences_vtrl(new_preferences))
        return pref


def clip_preferences_vtrl(preferences):
    # clip weights to be between 0 and 1, distribute teh rest between
    remainder = 0.0
    # clip any weights to 1 and distribute teh reminder to the other weights
    for i in range(len(preferences)):
        if preferences[i] > 1.0:
            remainder += preferences[i] - 1.0
            preferences[i] = 1.0
    # distribute the reminder evenly between the other non one weights
    if remainder == 0.0:
        return preferences
    none_one = sum(preferences < 1.0)
    for i in range(len(preferences)):
        if preferences[i] < 1.0:
            preferences[i] += remainder / none_one

    return clip_preferences_vtrl(preferences)
