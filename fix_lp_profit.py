# Read the current file
with open('defi_insurance_core.py', 'r') as f:
    content = f.read()

# Find the LP profit function
old_lp_function = '''    def lp_profit(self, c_c: float, c_lp: float, c_spec: float,
                  p_hack: float, expected_lgh: float, gamma: float) -> float:'''

new_lp_function = '''    def lp_profit(self, c_c: float, c_lp: float, c_spec: float,
                  p_hack: float, expected_lgh: float, gamma: float, risk_compensation: float = 1.5) -> float:'''

content = content.replace(old_lp_function, new_lp_function)

# Find the LP profit return statement
old_lp_return = '''        return yield_share - expected_payout - opportunity_cost'''

new_lp_return = '''        # Add risk compensation for capital at risk
        # LPs should be compensated for the risk they bear beyond just yield
        coverage = self.coverage_function(c_c, self.tvl)
        capital_at_risk = min(c_lp, coverage)  # Amount that could be lost
        risk_compensation_value = capital_at_risk * self.params.rho * risk_compensation
        
        return yield_share - expected_payout - opportunity_cost + risk_compensation_value'''

content = content.replace(old_lp_return, new_lp_return)

# Write back
with open('defi_insurance_core.py', 'w') as f:
    f.write(content)

print("✅ Fixed LP profit function with risk compensation")
print("💰 LPs now get compensated for capital at risk")
