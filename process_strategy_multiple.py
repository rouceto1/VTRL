#!/usr/bin/env python3
from process_strategy import process
import os
pwd = os.getcwd()
experiments_path = os.path.join(pwd, "experiments")

if __name__ == "__main__":
    paths = ["one", "two","three","four","five","six","seven","eight"]
    process(paths)