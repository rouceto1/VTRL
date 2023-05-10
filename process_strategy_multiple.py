#!/usr/bin/env python3
from process_strategy import process



if __name__ == "__main__":

    paths = ["all", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    teach = False
    e_teach = False
    e_grade = False
    grade = True
    redo = [teach, e_teach, e_grade, grade]
    process(paths, redo)
