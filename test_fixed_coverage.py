from defi_insurance_core import InsuranceMarket, MarketParameters

# Monkey patch the broken coverage function
def fixed_coverage_function(self, c_c, tvl, security_factor=1.0):
    # Sensible coverage: max coverage is collateral * amplification factor
    return min(self.params.mu * c_c * security_factor, tvl * 0.5)  # Max 50% of TVL

# Apply the fix
InsuranceMarket.coverage_function = fixed_coverage_function

params = MarketParameters(mu=1.5, theta=0.7)
market = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c = 10_000_000
market.c_lp = 50_000_000

state = market.get_market_state()
print(f'Fixed Coverage: ${state["coverage"]:,.0f}')
print(f'Fixed Utilization: {state["utilization"]:.2f}')
print(f'Coverage ratio: {state["coverage"]/state["tvl"]:.1%}')
