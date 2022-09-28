'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

from __future__ import print_function
import os
import sys
print(sys.version)
# As its called from blender we need to add the following
# to import from inside project
PROJECT_PATH = os.path.abspath('..')
sys.path.insert(0, PROJECT_PATH)
print("Python paths:")
print(sys.path)
import random
import argparse
import json
import tempfile
import pycocotools.mask as mask_utils
from collections import Counter
import cv2
import numpy as np
from mathutils.bvhtree import BVHTree
