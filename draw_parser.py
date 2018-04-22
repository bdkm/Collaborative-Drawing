import numpy as np

def scale_and_center(inks, scale, center):
    def scale_center_stroke(ink):
        xs = np.array(ink[0::2])
        ys = np.array(ink[1::2])

        xs = ((xs) * scale[0]) - (scale[0] / 2) + center[0]
        ys = ((ys) * scale[1]) - (scale[1] / 2) + center[1]
        return [val for pair in zip(xs, ys) for val in pair]

    return [scale_center_stroke(ink) for ink in inks]
