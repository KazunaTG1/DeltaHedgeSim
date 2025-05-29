from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, log, exp
from statistics import NormalDist
import copy
import pandas as pd
from itertools import islice

# Black-Scholes model for European options pricing
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
    def call_delta(self):
        term2 = BlackScholes(self.stock_price + 1, self.strike, self.years, self.implied_vol, self.interest_rate)
        return term2.call_price() - self.call_price()

    # Numerical approximation of put delta
    def put_delta(self):
        term2 = BlackScholes(self.stock_price + 1, self.strike, self.years, self.implied_vol, self.interest_rate)
        return term2.put_price() - self.put_price()


# Represents a stock position
class StockPosition:
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity
        self.market_value = price * quantity

    # Numerical approximation of delta (change in value with $1 price move)
    def delta(self):
        term2 = StockPosition(self.price + 1, self.quantity)
        return term2.market_value - self.market_value



# Geometric Brownian Motion stock price simulation
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


# Snapshot of a delta hedge at a point in time
class DeltaHedge:
    def __init__(self, s0, k, t, iv, r, prev_dS):
        self.option = BlackScholes(s0, k, t, iv, r)
        self.op_pl_delta = -self.option.call_delta() * 100  # Short 1 call = -delta * 100 shares
        if t == 0:
            self.op_pl_delta = 0  # No delta hedge at expiration

        self.stock_position = StockPosition(self.option.stock_price, -self.op_pl_delta)
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


# Runs a full delta hedging simulation
class DeltaHedgeSim:
    def __init__(self, s0, k, t, iv, r):
        self.option = BlackScholes(s0, k, t, iv, r)
        self.premium_received = self.option.call_price() * 100

    # Simulate hedge adjustments along the price path
    def simulate(self, n_steps=200):
        stock_path = GeometricBrownianMotion.sequence(
            self.option.stock_price,
            self.option.years,
            self.option.implied_vol,
            self.option.interest_rate,
            n_steps
        )
        points = [DeltaHedge(self.option.stock_price, self.option.strike, self.option.years, self.option.implied_vol, self.option.interest_rate, 0)]
        for tau, price in islice(stock_path.items(), 1, None):
            hedge = DeltaHedge(price, self.option.strike, tau, self.option.implied_vol, self.option.interest_rate, points[-1].stock_position.delta())
            points.append(hedge)

        # Final adjustment at expiration
        last = list(stock_path.values())[-1]
        final = DeltaHedge(last, self.option.strike, 0, self.option.implied_vol, self.option.interest_rate, points[-1].stock_position.delta())
        points.append(final)
        return points, stock_path


# Set parameters and run the simulation
years = 0.2
sim = DeltaHedgeSim(100, 105, years, 0.3, 0.0412)
n_steps = int(years * 504)
hedges, stock_path = sim.simulate(n_steps)

# Summarize the results
res = []
total_cashflow = 0
for h in hedges:
    res.append(h.display_form())
    total_cashflow += h.cash_flow

option_payout = sim.premium_received - (hedges[-1].option.call_price() * 100)
start_stock_price = hedges[0].option.stock_price
starting_cost = -hedges[0].cash_flow + sim.premium_received
pl = total_cashflow + option_payout

# Formats title with hyphens
def formatted_title(title, char='-'):
    star_len = int((30 - len(title)) / 2)
    offset = 0
    if len(title) % 2 == 0:
        offset = -1
    l_len = star_len + offset
    return f"{char * l_len} {title} {char * star_len}"
# Output P/L summary
print(f"Starting Cost: ${starting_cost:.2f}")
print(formatted_title("STOCK DETAILS"))
print(f"Starting Price: ${hedges[0].option.stock_price:.2f}")
print(f"Final Price: ${hedges[-1].option.stock_price:.2f}")
print(f"Total Cashflow: ${total_cashflow:.2f}")
print(formatted_title("OPTION DETAILS"))
print(f"Start Price: ${hedges[0].option.call_price():.2f}")
print(f"Final Price: ${hedges[-1].option.call_price():.2f}")
print(f"Premium Received: ${sim.premium_received:.2f}")
print(f"Option Payout: ${option_payout:.2f}")
print(formatted_title("PROFIT / LOSS"))
print(f"P/L: ${pl:.2f}")
print(f"Return on Cost: {pl / starting_cost*100:.2f}%")


# Display results
df = pd.DataFrame(res)
df.head()
df.style.hide(axis='index')
