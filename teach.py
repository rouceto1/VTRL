from python.evaluate import *
from python.general_paths import *
import src.NN_new.train_siam as nn_train


def teach(_dataset_path, _chosen_positions, _experiments_path,init_weights=None, _cache=None, conf=None):
    file_list = make_combos_for_teaching(_chosen_positions, _dataset_path, filetype_FM, conf=conf)
    files_with_displacement = fm_eval(file_list, filetype_FM)
    desired_files = np.array(choose_proper_filetype(filetype_NN, files_with_displacement))
    actual_teaching_count = nn_train.NNteach_from_python(desired_files, "strands", init_weights, _experiments_path, conf)
    return file_list, actual_teaching_count #TODO make it reaturn model and pass it to eval


if __name__ == "__main__":
    config = load_config("./NN_config_tests.yaml", 512)
    teach(dataset_path, chosen_positions, weights_file+".solo", cache, config)
