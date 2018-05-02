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

def add_jitter(stroke, jitter_amount):
    if len(stroke) == 0:
        return stroke
    zer = np.zeros(stroke.shape)
    r = np.array([[random.uniform(-jitter_amount, jitter_amount) for _ in range(2)] for _ in range(stroke.shape[0])])
    stroke[:,0:2] += r
    return stroke

def generate_wiggle(num_points):
    def random_coordinate(angle):
        distance = random.uniform(0.0,1.0)
        return [distance * math.cos(angle), distance * math.sin(angle), 0]
    angles = [np.random.normal(0,0.4) for _ in range(num_points)]
    angles = rolling_sum(angles)

    inks = np.array([random_coordinate(a) for a in angles])
    inks[-1][2] = 1
    return inks

def randomize_direction(stroke):
    if random.uniform(0.0, 1.0) < 0.5:
        stroke = stroke[::-1]
        if len(stroke) > 0:
            stroke[0][2] = 0.0
            stroke[-1][2] = 1
    return stroke

def format(inks):
    inks = ip.normalize_and_compute_deltas(rolling_sum(inks))
    print(inks)
    return inks

def cycle(inks, i):
    inks[-1][2] = 0
    a = inks[:i]
    b = inks[i:]
    inks = np.concatenate((b, a), axis=0)
    inks[-1][2] = 1
    return inks

def shear_x(inks, amount):
    inks[:,0] = inks[:,0] + amount * inks[:,1]
    return inks

def shear_y(inks, amount):
    inks[:,1] = inks[:,1] + amount * inks[:,0]
    return inks

def bow_x(inks, amount):
    inks[:,0] = ((inks[:,0]-0.5) * ((1-amount)  + (amount*inks[:,1]))) + 0.5
    return inks

def bow_y(inks, amount):
    inks[:,1] = ((inks[:,1]-0.5) * ((1-amount)  + (amount*inks[:,0]))) + 0.5
    return inks


def generate():
    num_los = random.randint(0,6)
    class_index = num_los
    num_points = random.randint(12,100)
    if num_los > 0:
        num_points = int(num_points / (2 * num_los))
    wiggle = generate_wiggle(num_points)

    if num_los > 0:
        if random.uniform(0.0,1.0) < 1.5:
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

        # makes sure start is at random position
        #final = rotate_around_center(final, random.randint(1, 2 * num_los) * math.pi / num_los)
        if num_los > 1:
            index = random.randint(1, num_points * 2 * num_los)
            final = cycle(final, index)
    else:
        class_index = 0
        final = wiggle

    if (True):
        final = add_jitter(final, 0.1)

    final = rolling_sum(final)
    if (True):

        final = ip.normalise(final)
        final = shear_x(final,np.random.normal(-0.065,0.1))
        final = shear_y(final,np.random.normal(-0.06,0.1))
        final = bow_x(final, np.random.normal(0.1,0.1))
        final = bow_y(final, np.random.normal(0.1,0.1))
    final = randomize_direction(final)
    final = ip.compute_deltas(final)

    return format(final), class_index
