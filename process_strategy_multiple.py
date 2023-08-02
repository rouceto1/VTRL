#!/usr/bin/env python3
from process_strategy import process
import os


if __name__ == "__main__":
    pwd = os.getcwd()
    experiments_path = os.path.join(pwd, "experiments")
    paths = [item for item in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, item))]
    teach = False
    e_teach = False
    e_grade = False
    grade = False
    redo = [teach, e_teach, e_grade, grade]
    paths.sort()
    process(paths, redo)
