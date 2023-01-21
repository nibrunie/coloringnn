import sys
import argparse
from  glob import glob

import os
from os import listdir
from os.path import isfile, join, basename

import random
import collections # for defaultdict

import numpy as np
import cv2

# progress bar example, taken from
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
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

def augment(color_img):
    yield color_img
    hflip = cv2.flip(color_img, 1)
    yield hflip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coloring CNN')
    parser.add_argument("input_dir", type=str,
                       help="input directory")
    parser.add_argument("--output", type=str, default="new_dataset/",
                   help="dataset output directory")
    parser.add_argument("--dataset-size", type=int, default=-1,
                   help="size of the dataset subset to use during training")
    parser.add_argument("--augment", action="store_const", const=True, default=False,
                   help="enable dataset augmentation")
    parser.add_argument("--input-size",
                        type=(lambda s: tuple(map(int, s.split(',')))), default=(128, 128),
                        help="dataset training input dimension")
    parser.add_argument("--super-size",
                        type=(lambda s: tuple(map(int, s.split(',')))), default=(256, 256),
                        help="dataset output dimension (super size)")

    args = parser.parse_args()

    def is_image_file(f):
        _, ext = os.path.splitext(f)
        return isfile(f) and ext in [".jpg", ".JPG"]

    # checking existence of output directories
    outDirColor = join(args.output, "color")
    outDirSuper = join(args.output, "super")
    for reqDir in [args.input_dir]:
        if not os.path.isdir(reqDir):
            print(f"[ERROR] required directory {reqDir} not found")
            raise FileNotFoundError

    for dirToCreate in [outDirColor, outDirSuper]: 
        if not os.path.isdir(dirToCreate):
            os.mkdir(dirToCreate)

    expanded_img_candidates = glob(join(args.input_dir, "**", "*.JPG"), recursive=True) +  glob(join(args.input_dir, "**", "*.jpg"), recursive=True)

    input_images = [f for f in expanded_img_candidates if is_image_file(f)]
    print("{} image file(s) found.".format(len(input_images)))
    processed_imgs = collections.defaultdict(lambda: 0)
    subset_len = len(input_images) if args.dataset_size < 0 else args.dataset_size
    subset_len = min(len(input_images), subset_len)
    for index, img_abspath in enumerate(input_images[:subset_len]):
            img_path = basename(img_abspath)
            input_img = cv2.imread(img_abspath)
            resized_img = cv2.resize(input_img, args.super_size)
            print(f"generating for {img_path} with shape {resized_img.shape[0]}x{resized_img.shape[1]}")
            if args.augment:
                extended_img_list = augment(resized_img)
            else:
                extended_img_list = [resized_img]
            for color_img in extended_img_list:
                input_img = cv2.resize(color_img, args.input_size)
                # rename already existing filename
                if img_path in processed_imgs:
                    prefix, suffix = os.path.splitext(img_path)
                    count = processed_imgs[img_path]
                    img_path = "{}-{}{}".format(prefix, count, suffix)
                cv2.imwrite(join(outDirColor, img_path), input_img)
                cv2.imwrite(join(outDirSuper, img_path), color_img)
                printProgressBar(index, subset_len, length=50)
                processed_imgs[img_path] += 1


