import sys
import argparse
from  glob import glob

import os
from os import listdir
from os.path import isfile, join, basename

import random

import numpy as np
import cv2

# progress bar example, taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='', suffix='',
                    decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coloring CNN')
    parser.add_argument("input_dir", type=str,
                       help="input directory")
    parser.add_argument("--output", type=str, default="new_dataset/",
                   help="dataset output directory")
    parser.add_argument("--dataset-size", type=int, default=100,
                   help="size of the dataset subset to use during training")
    parser.add_argument("--resize",
                        type=(lambda s: map(int, s.split(','))), default=(128, 128),
                        help="resize dimension")

    args = parser.parse_args()

    def is_image_file(f):
        _, ext = os.path.splitext(f)
        return isfile(f) and ext in [".jpg", ".JPG"]

    input_images = [f for f in glob(join(args.input_dir, "**", "*.JPG"), recursive=True) if is_image_file(f)]
    print("{} image file(s) found.".format(len(input_images)))
    for index, img_abspath in enumerate(input_images[:args.dataset_size]):
            img_path = basename(img_abspath)
            input_img = cv2.imread(img_abspath)
            resized_img = cv2.resize(input_img, args.resize)
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(join(args.output, "color", img_path), resized_img)
            cv2.imwrite(join(args.output, "gray", img_path), gray_img)
            printProgressBar(index, args.dataset_size, length=50)


