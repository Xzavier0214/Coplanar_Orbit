from gym.envs.classic_control import rendering
import numpy as np
from numpy import pi, cos, sin, arctan2


def make_arrow(start, end):
    translation = start
    rotation = arctan2(end[1] - start[1], end[0] - start[0])

    l = np.linalg.norm(np.array(end) - np.array(start))

    line = rendering.make_polyline(((0, 0), (l, 0)))
    line.set_linewidth(2)
    line.add_attr(rendering.Transform(translation, rotation))

    x = l/6*cos(pi/6)
    y = l/6*sin(pi/6)
    line_u = rendering.make_polyline(((l, 0), (l - x, y)))
    line_d = rendering.make_polyline(((l, 0), (l - x, -y)))
    line_u.set_linewidth(2)
    line_d.set_linewidth(2)
    line_u.add_attr(rendering.Transform(translation, rotation))
    line_d.add_attr(rendering.Transform(translation, rotation))

    geom = rendering.Compound([line, line_u, line_d])
    return geom


if __name__ == "__main__":
    viewer = rendering.Viewer(600, 400)

    arrow = make_arrow((100, 100), (200, 200))
    viewer.add_geom(arrow)

    for _ in range(40):
        viewer.render()
