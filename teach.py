from python.evaluate import *
from python.general_paths import *
import src.NN_new.train_siam as nn_train


def teach(_dataset_path, _mission, init_weights=None, _cache=None, conf=None):
    # make cache for files with displacement
    if _cache is None and conf["use_cache"]:
        _cache = "/tmp/cache.pkl"
    if conf["use_cache"]:
        if os.path.exists(_cache):
            with open(_cache, 'rb') as handle:
                files_with_displacement = pickle.load(handle)
        else:
            files_with_displacement = fm_eval(_mission.c_strategy.file_list, filetype_FM)
            with open(_cache, 'wb') as handle:
                pickle.dump(files_with_displacement, handle)
    else:
        files_with_displacement = fm_eval(_mission.c_strategy.file_list, filetype_FM)
    actual_teaching_count = nn_train.NNteach_from_python(files_with_displacement, "strands", init_weights,
                                                         _mission, conf)
    return actual_teaching_count  # TODO make it reaturn model and pass it to eval


if __name__ == "__main__":
    config = load_config("./NN_config_tests.yaml", 512)
    # teach(dataset_path, chosen_positions, weights_file+".solo", cache, config)
