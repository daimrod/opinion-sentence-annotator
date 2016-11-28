#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Base imports
import sys
import os
import inspect
import argparse


def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False):  # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)

sys.path.append(get_script_dir())

import logging


if 'logger' not in locals():
    logger = logging.getLogger('__run__')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # FileHandler
    fh = logging.FileHandler('log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

# User imports
from cnn import CNNBase, CNNChengGuo


# User functions
def main():
    parser = argparse.ArgumentParser(description='CNN runner.')
    parser.add_argument('-nb_epoch',  type=int, default=2,
                        help='The number of epoch.')
    parser.add_argument('-m', '--message', type=str,
                        help='A message to log at the start.')
    args = parser.parse_args()
    logger.info(args.message)
    try:
        CNNChengGuo(nb_epoch=args.nb_epoch).run()
    except Exception as ex:
        logger.error(ex)

if __name__ == '__main__':
    main()
