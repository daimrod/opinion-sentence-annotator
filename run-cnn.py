#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Base imports
import sys
import os
import inspect
import argparse
import ast


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
import logger_config

logger = logging.getLogger('__run__')

# User imports
from cnn import CNNBase
from cnn import CNNChengGuo
from cnn import CNNRegister


# User functions
def main():
    parser = argparse.ArgumentParser(description='CNN runner.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--message', type=str, default='',
                        help='A message to log at the start.')
    parser.add_argument('--model', type=str, default='CNNBase', choices=CNNRegister,
                        help='The name of the model to use.')
    known_args = []
    for model_name in CNNRegister:
        model = CNNRegister[model_name]
        spec = inspect.getargspec(model.__init__)
        if spec.defaults is None:
            continue
        for arg, val in zip(spec.args[1:], spec.defaults):
            if arg in known_args:
                continue
            else:
                known_args.append(arg)
            if type(val) == str:
                parser.add_argument('-' + arg, type=str, default=val,
                                    help='default: %(default)s')
            else:
                parser.add_argument('-' + arg, type=ast.literal_eval, default=val,
                                    help='default: %(default)s')
    args = parser.parse_args()
    logger.info(args.message)
    logger.info(args)
    try:
        model = CNNRegister[args.model]
        model(**args.__dict__).run()
    except Exception as ex:
        logger.exception(ex)

if __name__ == '__main__':
    main()
