# Represents a stock position
class StockPosition:
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity
        self.market_value = price * quantity

    # Numerical approximation of delta (change in value with $1 price move)
    def delta(self, h=0.01):
        bumped = StockPosition(self.price + h, self.quantity)
        return (bumped.market_value - self.market_value) / h