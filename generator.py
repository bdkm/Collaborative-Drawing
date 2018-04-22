import random
import numpy as np
import ink_parser as ip
import math

def connect(stroke1, stroke2):
    stroke2 = stroke2[::-1]
    if len(stroke1) > 0:
        stroke1[-1][2] = 0
    if len(stroke2) > 0:
        stroke2[0][2] = 0
        stroke2[-1][2] = 1
    return np.concatenate([stroke1, stroke2])

def reverse(ink):
    ink = np.copy(ink)
    return ink[::-1]

def reflect_accross_center(stroke):
    reflected = np.copy(stroke)
    reflected[:,1] = 0 - reflected[:,1]
    return reflected

def rotate(p, o, angle):
    qx = o[0] + math.cos(angle) * (p[0] - o[0]) - math.sin(angle) * (p[1] - o[1])
    qy = o[1] + math.sin(angle) * (p[0] - o[0]) + math.cos(angle) * (p[1] - o[1])
    return [qx, qy, p[2]]

def rotate_around_center(inks, angle):
    inks = np.copy(inks)
    inks = rolling_sum(inks)
    rotated = np.array([rotate(p, [0.0,0.0], angle) for p in inks])
    return ip.compute_deltas(rotated)

def rolling_sum(l):
    for i in range(1, len(l)):
        l[i] = l[i - 1] + l[i]
    return l

def generate_wiggle(num_points):
    def random_coordinate(angle):
        distance = random.uniform(0.0,1.0)
        return [distance * math.cos(angle), distance * math.sin(angle), 0]
    angles = [random.uniform(-math.pi / 8, math.pi / 8) for _ in range(num_points)]
    angles = rolling_sum(angles)

    inks = np.array([random_coordinate(a) for a in angles])
    inks[-1][2] = 1
    return inks

def format(inks):
    inks = ip.normalize_and_compute_deltas(rolling_sum(inks))
    return inks#np.array(ip.ink_rep_to_array_rep(inks))

def generate():
    num_los = random.randint(0,6)
    class_index = num_los
    num_points = random.randint(12,100)
    if num_los > 0:
        num_points = int(num_points / (2 * num_los))
    wiggle = generate_wiggle(num_points)

    if num_los > 0:
        if random.uniform(0.0,1.0) > 0.5:
            # Reflect
            side = reflect_accross_center(wiggle)
            side = connect(wiggle, side)
            final = side
            for i in range(1,num_los):
                a = reverse(rotate_around_center(side, 2 * i * math.pi / num_los))
                final = connect(final, a)
        else:
            # Rotate
            class_index += 6
            side = reverse(rotate_around_center(wiggle, 2 * math.pi / (num_los + 1)))
            final = connect(wiggle, side)
            for i in range(1,num_los):
                a = reverse(rotate_around_center(wiggle, 2 * (i + 1) * math.pi / (num_los + 1)))
                final = connect(final, a)
    else:
        class_index = 0
        final = wiggle

    return format(final), class_index
