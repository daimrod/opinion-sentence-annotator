#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Base imports
import sys
import os
import inspect


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
from main import NRCCanada
from main import Custom0, Custom0_with_SVD
from main import Custom1, Custom1_with_SVD
from main import Custom2, Custom2_with_SVD


# User functions
def main():
    if len(sys.argv) == 2:
        logger.info(sys.argv[1])
    parameters = {'topn': 1000,
                  'n_components': 50}
    for model in [NRCCanada,
                  Custom0, Custom0_with_SVD,
                  Custom1, Custom1_with_SVD,
                  Custom2, Custom2_with_SVD]:
        try:
            logger.info('%s with %s' % (model.__name__, parameters))
            model(**parameters).run()
        except Exception as ex:
            logger.error(ex)

if __name__ == '__main__':
    main()
