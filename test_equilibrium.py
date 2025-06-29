from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=3.763, theta=0.241)
market = InsuranceMarket(params)
market.tvl = 100_000_000
solver = EquilibriumSolver(market)

print('Testing equilibrium solver...')
c_c, c_lp, c_spec = solver.find_equilibrium(max_iterations=10)
print(f'Results: C_C=${c_c:,.0f}, C_LP=${c_lp:,.0f}, C_spec=${c_spec:,.0f}')

# Test with manual values
market.c_c = 10_000_000
market.c_lp = 50_000_000
market.c_spec = 2_000_000
state = market.get_market_state()
print(f'Manual test - Coverage: ${state["coverage"]:,.0f}')
print(f'Manual test - Utilization: {state["utilization"]:.2f}')
