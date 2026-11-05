import json
import numpy as np
import KratosMultiphysics as Kratos


def read_sensors(path):
    return json.load(open(path))["list_of_sensors"]


def locate(model_part, sensors):
    """Containing element and barycentric weights per sensor. Computed once."""
    index = {n.Id: i for i, n in enumerate(model_part.Nodes)}
    located = []
    for s in sensors:
        px, py = s["location"][0], s["location"][1]
        for elem in model_part.Elements:
            nodes = list(elem.GetGeometry())
            x = np.array([n.X0 for n in nodes])
            y = np.array([n.Y0 for n in nodes])
            det = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            l1 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / det
            l2 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / det
            w = np.array([l1, l2, 1.0 - l1 - l2])
            if w.min() >= -1e-10:
                located.append((np.array([index[n.Id] for n in nodes]), w,
                                np.array(s["direction"], float)))
                break
        else:
            raise RuntimeError(f"sensor {s['name']} lies outside every element")
    return located


def displacements(model_part):
    return np.array([list(n.GetSolutionStepValue(Kratos.DISPLACEMENT))
                     for n in model_part.Nodes])


def read(field, located):
    return np.array([float((w @ field[idx]) @ d) for idx, w, d in located])


def interpolate(model_part, sensors):
    return read(displacements(model_part), locate(model_part, sensors))
