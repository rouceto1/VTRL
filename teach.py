from python.evaluate import *
from python.general_paths import *
import src.NN_new.train_siam as nn_train


def teach(_dataset_path, _chosen_positions, _experiments_path, _cache=None, conf=None):
    if not conf["use_cache"]:
        file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)
        files_with_displacement = fm_eval(file_list, filetype_FM)
        if conf["save_cache"] and _cache is not None:
            with open(_cache, 'wb') as handle:
                pickle.dump(files_with_displacement, handle)
            print("teach making new cache " + _cache)
    else:
        print("teach reading cache " + _cache)
        with open(_cache, 'rb') as handle:
            files_with_displacement = pickle.load(handle)
    desired_files = np.array(choose_proper_filetype(filetype_NN, files_with_displacement))
    mdl = nn_train.NNteach_from_python(desired_files, "strands", _experiments_path, conf)
    return file_list #TODO make it reaturn model and pass it to eval


if __name__ == "__main__":
    config = load_config("./NN_config_tests.yaml", 512)
    teach(dataset_path, chosen_positions, weights_file+".solo", cache, config)
