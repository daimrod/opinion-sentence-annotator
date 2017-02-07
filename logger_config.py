#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


logger = logging.getLogger(None)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
# StreamHandler
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

# FileHandler
fh = logging.FileHandler('log.txt', 'a')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)

logger.handlers = [sh, fh]
