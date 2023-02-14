from python.evaluate import *
from python.general_paths import *


def teach(_dataset_path, _chosen_positions, _weights_file, _cache, use_cache=False, limit=None):
    if not os.path.exists(_cache) or not use_cache:
        file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, limit=limit)
        print("evaluating FM")
        files_with_displacement = fm_eval(file_list, filetype_FM)
        with open(_cache, 'wb') as handle:
            pickle.dump(files_with_displacement, handle)
        print("making new cache " + _cache)
    else:
        print("reading cache " + _cache)
        with open(_cache, 'rb') as handle:
            files_with_displacement = pickle.load(handle)
    print("Predicaments acquired " + _cache)
    desired_files = np.array(choose_proper_filetype(filetype_NN, files_with_displacement))
    nn_ev.NNteach_from_python(desired_files, "strands", os.path.join(_dataset_path, _weights_file), 3)


if __name__ == "__main__":
    teach(dataset_path, chosen_positions, weights_file+".solo", cache, use_cache=False, limit=20)
