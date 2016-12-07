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

# User imports
import embeddings as emb
import resources as res
import features as feat
from reader import read_bing_liu

# User functions


def main():
    parser = argparse.ArgumentParser(description='Build inequalities.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output',
                        help='Output file')
    args = parser.parse_args()
    model = emb.get_custom0()
    lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                            res.bing_liu_lexicon_path['positive'])
    print('Save inequalities to %s', args.output)
    feat.find_ineq(model, lexicon, args.output)


if __name__ == '__main__':
    main()
