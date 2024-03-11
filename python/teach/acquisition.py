
from abc import ABC, abstractmethod
import os
from python.teach.planner import VTRE


class Robot(ABC):
    #Represents place wehre to gather data. Etiehr simulation, robot or dataset.
    @abstractmethod
    def make_combos_for_dataset(self, strategy,timetable, time_start, time_end):
        """
        Translate timetable to file list
        """

        raise NotImplementedError


class In_Dataset(Robot):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.fimage_file_template = "season_%04d/%09d"

    def make_combos_for_dataset(self,timetable, time_start, time_end):
        output, count = self.make_combos(timetable[0], os.path.join(self.dataset_path, "cestlice"),
                                            self.fimage_file_template)
        output2, count2 = self.make_combos(timetable[1], os.path.join(self.dataset_path, "strands"),
                                              self.fimage_file_template)
        output.extend(output2)
        return output, count + count2

    def make_combos(self, input, path_dataset, image_file_template):
        # makes list of all possible combination between same places per season
        # input is list of seasons which are folders
        # each season is list of places which are files
        # each file is defined as "season_%d04/%09d"
        # format of return is: [cestlice[season0[place1 place2 place3 ... place271 ] season1[place1 place2 place3 ... place271 ] ... season30[place1 place2 place3 ... place271 ]]
        #                                       strands[season0[place1 place2 place3 ... place7 ] season1[place1 place2 place3 ... place7 ] ... season1007[place1 place2 place3 ... place7 ]]]

        output = []
        suffix = ".png"
        seasons = len(input)
        places = len(input[0])
        # places are amount of folders in path_dataset
        count = 0
        for place in range(places):
            for season in range(seasons):
                if input[season][place] == 0:
                    continue
                for season2 in range(season + 1, seasons):
                    if input[season2][place] == 0:
                        continue
                    count += 1
                    file1 = os.path.join(path_dataset, image_file_template % (season, place)) + suffix
                    file2 = os.path.join(path_dataset, image_file_template % (season2, place)) + suffix
                    if os.path.exists(file1) and os.path.exists(file2):
                        output.append([file1, file2])
        return output, count


class In_Simulation(Robot):

    def __init__(self, dataset_path=None):
        self.map_path = "/home/rouceto1/.ros"
        self.dataset_path = dataset_path
    def make_combos_for_dataset(self, timetable, time_start, time_end):
        # format of timetable is: [cestlice[season0[place1 place2 place3 ... place271 ] season1[place1 place2 place3 ... place271 ] ... season30[place1 place2 place3 ... place271 ]]
        #                                       strands[season0[place1 place2 place3 ... place7 ] season1[place1 place2 place3 ... place7 ] ... season1007[place1 place2 place3 ... place7 ]]]
        # takes chosen postioins list and creates all possible teaching combinattaions for it
        # each combination is a list of 2 files comprised of same place between 2 seasons
        # available places are indicated by 1 not avialbale are 0

        self.acquire_new_data_from_timetable(timetable, time_start, time_end)
        output, count = self.make_combos(self.map_path, self.dataset_path, "mall_vtr")
        output2, count2 = self.make_combos(self.map_path, self.dataset_path, "forest_vtr")
        output.extend(output2)
        return output, count + count2


    # makes the file list from all images to all targets images
    def make_file_list_annotation(self, places, images, evaluation_prefix, evaluation_paths, target=112):
        combination_list2 = []
        file_list2 = []
        for place in places:  # range(7):
            for nmbr in images:  # range(1,143):
                combination_list2.append([place, target, nmbr])
        for e in evaluation_paths:
            for combo in combination_list2:
                file1 = os.path.join(evaluation_prefix, evaluation_paths[0], self.image_file_template % (combo[0], combo[1]))
                file2 = os.path.join(evaluation_prefix, e, self.image_file_template % (combo[0], combo[2]))
                if os.path.exists(file1 + ".png") and os.path.exists(file2 + ".png"):
                    file_list2.append([file1, file2])
        return file_list2

    def acquire_new_data_from_timetable(self, timetable, time_start, time_end):
        # TODO run robot over the required path and save the data properly using the simulator
        simulator = VTRE()
        assert time_end <=len(timetable)
        for tenner in range(int(time_start), int(time_end)):
            if True in timetable[tenner]:
                for map in timetable[tenner]:
                    print("Running simulator for map: ", map, " at time: ", tenner)
                    simulator.run(tenner/144.0, map, tenner)




    def make_combos(self, map_path, dataset_path, map_name):
        # get all folders in dataset_path corespondong to specific map_name
        all_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir() and map_name in f.path]
        ##all_folders.append(map_path)
        print(all_folders)
        # walk each folder and matach all available images to other images using distance
        pairs = []
        for idx, folder1 in enumerate(all_folders):
            for folder2 in all_folders[idx:]:
                if folder1 == folder2:
                    continue
                pairs.extend(self.match_distances_in_two_folders(folder1, folder2))
        return pairs, len(pairs)

    def match_distances_in_two_folders(self, folder1, folder2):
        # find files that are closest to each other in two folders
        # list all files in folder1:
        suffix = ".jpg"
        files1 = [f.path for f in os.scandir(folder1) if f.is_file() and suffix in f.path]
        files2 = [f.path for f in os.scandir(folder2) if f.is_file() and suffix in f.path]
        if len(files1) < len(files2):
            files1, files2 = files2, files1
        # extract numbers from file names removing suffix
        numbers1 = [int(f.split("/")[-1].split(".")[0]) for f in files1]
        numbers2 = [int(f.split("/")[-1].split(".")[0]) for f in files2]
        # find closest numbers

        pairs = self.find_closest(numbers1, numbers2)
        #return file pairs corresponding to the numbers
        return [[files1[i[0]], files2[i[1]]] for i in pairs]



    def find_closest(self, array1, array2, limit=2):
        pairs = []
        for i, num1 in enumerate(array1):
            closest = min(array2, key=lambda x: abs(x - num1))
            index = array2.index(closest)
            if num1 - closest < limit:
                continue
            pairs.append([i, index])
        return pairs

