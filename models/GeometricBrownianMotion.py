from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, log, exp
from statistics import NormalDist
import copy
import pandas as pd
from itertools import islice
import statistics
class GeometricBrownianMotion:
    # Generate next price point
    def next_term(s0, dt, iv, r, size=1):
        if size == 1:
            Z = np.random.normal()
            return s0 * exp((r - 0.5 * iv ** 2) * dt + iv * sqrt(dt) * Z)
        else:
            return [GeometricBrownianMotion.next_term(s0, dt, iv, r) for _ in range(size)]

    # Generate full price path
    def sequence(s0, t, iv, r, n_steps=100, size=1):
        dt = t / n_steps
        if size == 1:
            vals = {t: s0}
            for i in range(1, n_steps):
                last = list(vals.values())[-1]
                elapsed = i * dt
                tau = t - elapsed
                vals[tau] = GeometricBrownianMotion.next_term(last, dt, iv, r)
            return vals
        else:
            return [GeometricBrownianMotion.sequence(s0, t, iv, r, n_steps) for _ in range(size)]