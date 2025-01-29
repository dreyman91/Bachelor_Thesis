######### CLASS ADMIN ##########
class Admin:
    def __init__(self, market_env, max_price_factor = 1.5, penalty=3):
        self.market_env = market_env
        self.violations = {}
        self.vendor_penalties = {}
        self.max_price_factor = max_prcie_factor
        self.penalty = penalty

    def observe_market(self):
        return {
            "overall_market_trend": self.analyze_trends(),
            "vendor_status": {v: self.market_env.vendors_status[v] for v in self.market_env.vendors_status},
            "price_distributions": self.market_env.market_prices,
            "total_transactions": sum(len(v) for v in self.market_env.vendors_products.values()),
            "violations_detected": len(self.violations)
        }

    def analyze_trends(self):
        if not self.market_env.market_price_history:
            return 0
        last_prices = [history[-3:] for history in self.market_env.market_price_history.values()
                       if len(history) >=3]
        if not last_prices:
            return 0
        last_prices = np.array(last_prices)
        avg_trend = np.mean(last_prices[:, -1] - last_prices[:, 0])
        return np.sign(avg_trend)

    def fine(self):
        for vendor, products in self.market_env.vendors_products.items():
            for product in products:
                avg_market_price = np.mean(self.market_env.market_prices)
                price_threshold = avg_market_price * self.max_price_factor

                if product.price > price_threshold:
                    self.violations[vendor] = self.violations.get(vendor, 0) + 1
                    self.vendor_penalties[vendor] = self.penalty
                    self.market_env.vendors_status[vendor] = self.penalty
                    print(f"Admin penalized Vendor: {vendor}) for price gouging on {product}")

    def update_vendor_status(self):
        """ Reduce penalized vendor for 3 steps"""
        for vendor in self.vendor_penalties:
            if self.vendor_penalties[vendor] > 0:
                self.vendor_penalties[vendor] -= 1
                if self.vendor_penalties[vendor] == 0:
                    self.market_env.vendors_status[vendor] = "active"
                    print(f"{vendor} is active again after penalty")

    def enforce_policies(self):
        """Apply regulatory actions based on market conditions"""
        self.fine()
        self.update_vendor_status()