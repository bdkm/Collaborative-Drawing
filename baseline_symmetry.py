import tensorflow as tf
import numpy as np
import ink_parser as ip
import math

def rotate(p, o, angle):
    qx = o[0] + math.cos(angle) * (p[0] - o[0]) - math.sin(angle) * (p[1] - o[1])
    qy = o[1] + math.sin(angle) * (p[0] - o[0]) + math.cos(angle) * (p[1] - o[1])
    return [qx, qy, p[2]]

def rolling_sum(l):
    for i in range(1, len(l)):
        l[i] = l[i - 1] + l[i]
    return l

def analyse(inks):
    inks = np.array(ip.array_rep_to_ink_rep(inks))[0]
    inks = ip.normalise(inks)

    length = inks.shape[0]
    segment = int(length / 5)
    sum = 0.0
    for i in range(0,5):
        angle = -i * 2 * math.pi / 5

        inks_rotated = np.array([rotate(c,[0.5,0.5],angle) for c in inks])

        print("---")
        offset = segment * i
        a = inks_rotated[offset:]
        b = inks_rotated[:offset]
        if (i > 0):
            a[-1,2] = 0
            b[-1,2] = 1
            cycled = np.vstack((a,b))
        else:
            cycled = inks_rotated
        reversed = cycled[::-1]

        xs = cycled[:,0]
        ys = cycled[:,1]
        xn = -1 * reversed[:,0]
        yn = reversed[:,1]

        xdisp = abs(np.mean(np.array_split(np.add(xs, xn),2)[0]))
        print(xdisp)
        ydisp = abs(np.mean(np.array_split(np.add(ys, yn),2)[0]))
        print(ydisp)
        sum += xdisp
        sum += ydisp
        print(sum)
    sum
    print(sum)
