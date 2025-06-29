from defi_insurance_core import InsuranceMarket, MarketParameters
import numpy as np

params = MarketParameters(mu=1.5, theta=0.7)
market = InsuranceMarket(params)

print("Debug coverage function:")
print(f"mu: {params.mu}")
print(f"theta: {params.theta}")
print(f"TVL: {market.tvl}")

c_c = 10_000_000
tvl = 100_000_000

# Manual calculation of coverage function
# coverage = µ · C_C^θ · (1 + ξ) · TVL
coverage_manual = params.mu * (c_c ** params.theta) * (1 + params.xi) * tvl

print(f"C_C: ${c_c:,}")
print(f"C_C^theta: {c_c ** params.theta:,.0f}")
print(f"Manual coverage: ${coverage_manual:,.0f}")

# Test the actual function
coverage_function = market.coverage_function(c_c, tvl)
print(f"Function coverage: ${coverage_function:,.0f}")
