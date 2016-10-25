#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


if 'logger' not in locals():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # FileHandler
    fh = logging.FileHandler('run_log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

from main import runNRCCanada
from main import runCustom0, runCustom0_with_SVD
from main import runCustom1, runCustom1_with_SVD

runNRCCanada()
runCustom0()
runCustom0_with_SVD()
runCustom1()
runCustom1_with_SVD()
