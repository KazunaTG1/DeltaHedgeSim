from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, log, exp
from statistics import NormalDist
import copy
import pandas as pd
from itertools import islice
import statistics
class BlackScholes:
    def __init__(self, s0, k, t, iv, r):
        self.stock_price = s0
        self.strike = k
        self.years = t
        self.implied_vol = iv
        self.interest_rate = r

    # Copy method for modifying instances without affecting the original
    def __copy__(self):
        return BlackScholes(self.stock_price, self.strike, self.years, self.implied_vol, self.interest_rate)

    # Calculate d1 and d2 (standardized variables for Black-Scholes)
    def get_d1_d2(self):
        log_moneyness = log(self.stock_price / self.strike)
        risk_adj_return = self.interest_rate + 0.5 * self.implied_vol ** 2
        d1 = (log_moneyness + risk_adj_return * self.years) / (self.implied_vol * sqrt(self.years))
        d2 = d1 - (self.implied_vol * sqrt(self.years))
        return d1, d2

    # CDF values for call pricing
    def call_norms(self):
        d1, d2 = self.get_d1_d2()
        return NormalDist().cdf(d1), NormalDist().cdf(d2)

    # CDF values for put pricing
    def put_norms(self):
        d1, d2 = self.get_d1_d2()
        return NormalDist().cdf(-d1), NormalDist().cdf(-d2)

    # Black-Scholes call price formula
    def call_price(self):
        if self.years == 0: return max(0, self.stock_price - self.strike)
        norm_d1, norm_d2 = self.call_norms()
        return (self.stock_price * norm_d1) - (self.strike * self.discount_factor() * norm_d2)

    # Risk-free discount factor
    def discount_factor(self):
        return exp(-self.interest_rate * self.years)

    # Black-Scholes put price formula
    def put_price(self):
        if self.years == 0: return max(0, self.strike - self.stock_price)
        norm_d1, norm_d2 = self.put_norms()
        return (self.strike * self.discount_factor() * norm_d2) - (self.stock_price * norm_d1)

    # Numerical approximation of call delta
    def call_delta(self, h=0.01):
        bumped = BlackScholes(self.stock_price + h, self.strike, self.years, self.implied_vol, self.interest_rate)
        return (bumped.call_price() - self.call_price())/h

    # Numerical approximation of put delta
    def put_delta(self, h=0.01):
        bumped = BlackScholes(self.stock_price + h, self.strike, self.years, self.implied_vol, self.interest_rate)
        return (bumped.put_price() - self.put_price()) / h