from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, log, exp
from statistics import NormalDist
import copy
import pandas as pd
from itertools import islice
import statistics
from .. import BlackScholes as bs, StockPosition as sp
# Snapshot of a delta hedge at a point in time
class DeltaHedge:
    def __init__(self, s0, k, t, iv, r, prev_dS):
        self.option = bs.BlackScholes(s0, k, t, iv, r)
        self.op_pl_delta = -self.option.call_delta() * 100  # Short 1 call = -delta * 100 shares
        if t == 0:
            self.op_pl_delta = 0  # No delta hedge at expiration

        self.stock_position = sp.StockPosition(self.option.stock_price, -self.op_pl_delta)
        self.diff = self.stock_position.delta() - prev_dS  # Change in delta since last rebalance
        self.cash_flow = -(s0 * self.diff)                 # Cost of adjusting stock position
        self.net_exposure = round(self.stock_position.delta() + self.op_pl_delta, 5)

    # Returns results in dict form for display/logging
    def display_form(self):
        return {
            'Years to Maturity': self.option.years,
            'Call Price': self.option.call_price(),
            'Stock Price': self.stock_position.price,
            'Short Call Delta': self.op_pl_delta,
            'Stock Delta': self.stock_position.delta(),
            'Net Delta': self.net_exposure,
            'Position Change': self.diff,
            'Cash Flow': self.cash_flow
        }