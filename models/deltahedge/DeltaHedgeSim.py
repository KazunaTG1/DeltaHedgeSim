from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, log, exp
from statistics import NormalDist
import copy
import pandas as pd
from itertools import islice
import statistics
from .. import BlackScholes as bs, StockPosition as sp, GeometricBrownianMotion as gbm
from . import DeltaHedge as dh
# Runs a full delta hedging simulation
class DeltaHedgeSim:
    def __init__(self, s0, k, t, iv, r):
        self.option = bs.BlackScholes(s0, k, t, iv, r)
        self.premium_received = self.option.call_price() * 100

    # Simulate hedge adjustments along the price path
    def simulate(self, n_steps=200):
        stock_path = gbm.GeometricBrownianMotion.sequence(
            self.option.stock_price,
            self.option.years,
            self.option.implied_vol,
            self.option.interest_rate,
            n_steps
        )
        points = [dh.DeltaHedge(self.option.stock_price, self.option.strike, self.option.years, self.option.implied_vol, self.option.interest_rate, 0)]
        for tau, price in islice(stock_path.items(), 1, None):
            hedge = dh.DeltaHedge(price, self.option.strike, tau, self.option.implied_vol, self.option.interest_rate, points[-1].stock_position.delta())
            points.append(hedge)
            

        # Final adjustment at expiration
        last = list(stock_path.values())[-1]
        final = dh.DeltaHedge(last, self.option.strike, 0, self.option.implied_vol, self.option.interest_rate, points[-1].stock_position.delta())
        points.append(final)
        return points, stock_path