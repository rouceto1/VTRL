from python.evaluate import *
from python.general_paths import *
import src.NN_new.train_siam as nn_train


def teach(_dataset_path, _mission, init_weights=None, _cache=None, conf=None):
    # make cache for files with displacement
    if _cache is None and conf["use_cache"]:
        _cache = "/tmp/cache.pkl"
    if conf["use_cache"]:
        #TODO make possible to run for several conescutive runs
        if os.path.exists(_cache):
            with open(_cache, 'rb') as handle:
                files_with_displacement = pickle.load(handle)
        else:
            if _mission.c_strategy.iteration == 0:
                files_with_displacement = fm_eval(_mission.c_strategy.file_list, filetype_FM)
            with open(_cache, 'wb') as handle:
                pickle.dump(files_with_displacement, handle)
    else:
        if _mission.c_strategy.iteration == 0:

            files_with_displacement = fm_eval(_mission.c_strategy.file_list, filetype_FM)
        else:
            file_list, displacements, feature_count_l, feature_count_r, _, _, _ = fm_nn_eval(_mission.c_strategy.file_list,
                                                                                                      filetype_NN,
                                                                                                      filetype_FM, None,
                                                                                                      init_weights,
                                                                                                      None,
                                                                                                      conf)
            fm_out = np.array([displacements, feature_count_l, feature_count_r], dtype=np.float32).T
            files_with_displacement = np.append(file_list, fm_out, axis=1)

    actual_teaching_count = nn_train.NNteach_from_python(files_with_displacement, "strands", init_weights,
                                                         _mission, conf)
    return actual_teaching_count  # TODO make it reaturn model and pass it to eval


if __name__ == "__main__":
    config = load_config("./NN_config_tests.yaml", 512)
    # teach(dataset_path, chosen_positions, weights_file+".solo", cache, config)
