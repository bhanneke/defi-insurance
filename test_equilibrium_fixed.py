from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=1.5, theta=0.7, alpha=0.6, u_target=0.8)
market = InsuranceMarket(params)
market.tvl = 100_000_000

solver = EquilibriumSolver(market)
print('Testing equilibrium solver with fixed coverage...')
c_c, c_lp, c_spec = solver.find_equilibrium(max_iterations=50)
print(f'Equilibrium: C_C=${c_c:,.0f}, C_LP=${c_lp:,.0f}, C_spec=${c_spec:,.0f}')

market.c_c, market.c_lp, market.c_spec = c_c, c_lp, c_spec
state = market.get_market_state()
print(f'Final Coverage: ${state["coverage"]:,.0f}')
print(f'Final Utilization: {state["utilization"]:.4f}')
print(f'Revenue Share: {state["revenue_share"]:.3f}')
