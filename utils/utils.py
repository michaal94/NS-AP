import sys
import os
import shutil
from datetime import datetime

class CyclicBuffer:
    def __init__(self, buffer_size) -> None:
        self.buffer = []
        self.buffer_size = buffer_size
        self.pointer = 0

    def append(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.pointer] = item
            self.pointer = (self.pointer + 1) % self.buffer_size
        else:
            self.buffer.append(item)
        # print(self.buffer)

    def get(self):
        return self.buffer[self.pointer:] + self.buffer[:self.pointer]

    def __len__(self):
        return len(self.buffer)

    def flush(self):
        self.buffer = []
        self.pointer = 0

class ItemCarousel:
    def __init__(self, item_list=[]) -> None:
        self.item_list = item_list
        self.item_counter = 0

    def get(self):
        item = self.item_list[self.item_counter]
        self.item_counter = (self.item_counter + 1) % len(self.item_list)
        return item


def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--". Blender ignores command-line flags
    after --, so this lets us forward command line arguments from the blender
    invocation to our own script.
    """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv

# Make directories if path does not exists
# Input can be one path or list
def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


# Simple file copying
def copy_file(src, dst):
    shutil.copyfile(src, dst)


# Copy directories (with content)
def copy_dirs(src_list, dst):
    for d in src_list:
        shutil.copytree(d, os.path.join(dst, os.path.basename(d)), symlinks=True)


# Invert dictionary
def invert_dict(d):
    return {v: k for k, v in d.items()}


# Add timestamp to the name
def timestamp_dir(logdir):
    main_dir, exp_dir = os.path.split(logdir)
    # Append 'timestamp' to the experiment directory name
    now = datetime.now()
    yy = now.year % 100
    m = now.month
    dd = now.day
    hh = now.hour
    mm = now.minute
    ss = now.second
    timestamp = "{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(dd, m, yy, hh, mm, ss)
    exp_dir = "{}_{}".format(exp_dir, timestamp)
    logdir = os.path.join(main_dir, exp_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir