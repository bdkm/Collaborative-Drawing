import json as json
import numpy as np
"""
Functions for parsing a json line and preprocessing the ink record inside it.
"""

""" Parse json line and return class_name and ink records """
def parse_json(line):
    # Parse line as json and extract relevant fields
    sample = json.loads(line)
    class_name = sample["word"]
    inkarray = sample["drawing"]
    return class_name, inkarray

""" Fill numpy array with ink data """
def reshape_ink(inkarray):
    # Set up data
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    ink = np.zeros((total_points, 3), dtype=np.float32)
    # Fill ink matrix
    current_t = 0
    for stroke in inkarray:
        for i in [0, 1]:
            ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        ink[current_t - 1, 2] = 1
    return ink

"""Normalize between 0 and 1"""
def normalise(ink):
    # Normalize size to between 0 and 1
    lower = np.min(ink[:, 0:2], axis=0)
    upper = np.max(ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    ink[:, 0:2] = (ink[:, 0:2] - lower) / scale
    return ink

"""Scales"""
def scale(ink, scale_x, scale_y):
    ink[:,0] = (ink[:,0] + 1) * scale_x / 2
    ink[:,1] = (ink[:,1] + 1) * scale_y / 2
    return ink

"""Converts numpy array of coordinates into deltas"""
def compute_deltas(ink):
    ink[1:, 0:2] = ink[1:, 0:2] - ink[0:-1, 0:2]
    return ink

""" Normalize the ink between 0 and 1 and computes the difference between
    points (then between -1 and 1) """
def normalize_and_compute_deltas(ink):
    # Normalize size to between 0 and 1
    if len(ink) == 0:
        return ink
    lower = np.min(ink[:, 0:2], axis=0)
    upper = np.max(ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    ink[:, 0:2] = (ink[:, 0:2] - lower) / scale
    # Change to difference between points
    ink[1:, 0:2] = ink[1:, 0:2] - ink[0:-1, 0:2]
    return ink

""" Helper function for parsing an element """
def parse_element(line):
    class_name, ink = parse_json(line)
    ink = reshape_ink(ink)
    ink = normalize_and_compute_deltas(ink)
    return ink, class_name

def array_rep_to_ink_rep(ink):
    def reshape_2d(list):
        list = np.array(list)
        return list.reshape(-1,2)

    ink = [reshape_2d(stroke) for stroke in ink]

    def stroke_delimiter(height):
        zeros = np.zeros((height,1))
        zeros[-1] = 1
        return zeros

    ink = [np.hstack((stroke,stroke_delimiter(stroke.shape[0]))) for stroke in ink]
    return ink

def ink_rep_to_array_rep(ink):
    def split_by_indicies(l,ss):
        ll = []
        start = 0
        for s in ss:
            ll.append(l[start:s + 1])
            start = s + 1
        return np.array(ll)

    def rolling_sum(l):
        for i in range(1, len(l)):
            l[i] = l[i - 1] + l[i]
        return l

    def interleave(a,b):
        return [val for pair in zip(a, b) for val in pair]
    xs = ink[:,0]
    ys = ink[:,1]
    ss = ink[:,2]
    ss = np.nonzero(ss)[0]

    xs = rolling_sum(xs)
    ys = rolling_sum(ys)

    xs = split_by_indicies(xs, ss)
    ys = split_by_indicies(ys, ss)
    
    return [interleave(*pair) for pair in zip(xs,ys)]
