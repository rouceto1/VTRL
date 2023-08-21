#!/usr/bin/env python3
from process_strategy import process_old
import os


if __name__ == "__main__":
    pwd = os.getcwd()
    exp_path = "experiments"
    exp_path = "backups/mission"
    experiments_path = os.path.join(pwd, exp_path)
    paths = [item for item in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, item))]
    teach = False
    e_teach = False
    e_grade = False
    grade = True
    redo = [teach, e_teach, e_grade, grade]
    paths.sort()
    process_old(paths, exp_folder_name=exp_path)
