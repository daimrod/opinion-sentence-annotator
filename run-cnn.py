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
    parser = argparse.ArgumentParser(description='CNN runner.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--message', type=str,
                        help='A message to log at the start.')
    parser.add_argument('--model', type=str,
                        help='The name of the model to use.')
    for model in [CNNBase, CNNChengGuo]:
        spec = inspect.getargspec(model.__init__)
        for arg, val in zip(spec.args[1:], spec.defaults):
            parser.add_argument('-' + arg, type=type(val), default=val,
                                help='default: %(default)s')
    args = parser.parse_args()
    logger.info(args.message)
    logger.info(args)
    try:
        CNNChengGuo(**args.__dict__).run()
    except Exception as ex:
        logger.error(ex)

if __name__ == '__main__':
    main()
