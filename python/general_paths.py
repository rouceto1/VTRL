import argparse

from .helper_functions import *

pwd = os.getcwd()
parser = argparse.ArgumentParser(
    description='example: --dataset_path "full path" --evaluation_prefix "full path" --weights_folder "full path" '
                '--file_out suffix.picke')
parser.add_argument('--dataset_path', type=str,
                    help="full path to dataset to be annotated",
                    default="datasets/strands_crop/training_Nov")
parser.add_argument('--evaluation_prefix', type=str, help="path to folder with evaluation sub-folders",
                    default="datasets/strands_crop")
parser.add_argument('--evaluation_paths', type=str, help="names of folders to eval",
                    default=["testing_Dec", "testing_Feb", "testing_Nov"])
parser.add_argument('--weights_folder', type=str, help="path of weights folder",
                    default="weights/")
parser.add_argument('--weights_file', type=str, help="name of weights.pt",
                    default="model_eunord.pt")
parser.add_argument('--file_out', type=str, help="name of pickle out",
                    default="_GT_.pickle")
args = parser.parse_args()

dataset_path = os.path.join(pwd, args.dataset_path)
evaluation_prefix = os.path.join(pwd, args.evaluation_prefix)
evaluation_paths = args.evaluation_paths
annotation_file = args.weights_file
GT_file = annotation_file + args.file_out
weights_file = os.path.join(pwd, args.weights_folder) + annotation_file
neural_network_file = annotation_file + "_NN_cache.pickle"
cache2 = os.path.join(dataset_path, neural_network_file)
eval_out_file = weights_file + "_eval.pickle"

chosen_positions = np.loadtxt(os.path.join(dataset_path, chosen_positions_file), int)
eval_out = os.path.join(dataset_path, eval_out_file)
cache = os.path.join(dataset_path, feature_matcher_file)
