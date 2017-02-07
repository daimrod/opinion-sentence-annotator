#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


logger = logging.getLogger(__name__)

from base import preprocess
import resources as res

preprocess(res.train_path, force=True)
preprocess(res.test_path, force=True)
preprocess(res.dev_path, force=True)
