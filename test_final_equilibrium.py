import sys
if 'defi_insurance_core' in sys.modules:
    del sys.modules['defi_insurance_core']

from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

print("🎯 Testing Final Equilibrium with Fixed Profit Functions")
print("=" * 55)

solver = EquilibriumSolver(market)

# Test the fixed best response functions
print("Testing individual best responses:")
c_lp_test = 15_000_000
c_spec_test = 3_000_000

optimal_c_c = solver.protocol_best_response(c_lp_test, c_spec_test)
print(f"Protocol best response: ${optimal_c_c:,.0f}")

optimal_c_lp = solver.lp_best_response(optimal_c_c, c_spec_test)
print(f"LP best response: ${optimal_c_lp:,.0f}")

# Test full equilibrium
print(f"\nRunning full equilibrium solver:")
c_c_eq, c_lp_eq, c_spec_eq = solver.find_equilibrium(max_iterations=20)
print(f"Equilibrium: C_C=${c_c_eq:,.0f}, C_LP=${c_lp_eq:,.0f}, C_spec=${c_spec_eq:,.0f}")

# Test the equilibrium economics
market.c_c, market.c_lp, market.c_spec = c_c_eq, c_lp_eq, c_spec_eq
state = market.get_market_state()

p_hack = 0.1
expected_lgh = 0.1
protocol_profit = market.protocol_profit(c_c_eq, c_lp_eq, c_spec_eq, p_hack, expected_lgh, state['revenue_share'])
lp_profit = market.lp_profit(c_c_eq, c_lp_eq, c_spec_eq, p_hack, expected_lgh, state['revenue_share'])

print(f"\nEquilibrium Results:")
print(f"  Coverage: ${state['coverage']:,.0f} ({state['coverage']/state['tvl']:.1%} of TVL)")
print(f"  Utilization: {state['utilization']:.4f}")
print(f"  Revenue Share: {state['revenue_share']:.3f}")
print(f"  Protocol Profit: ${protocol_profit:,.0f}")
print(f"  LP Profit: ${lp_profit:,.0f}")
print(f"  Both profitable: {protocol_profit > 0 and lp_profit > 0}")

if protocol_profit > 0 and lp_profit > 0 and state['coverage'] > 1_000_000:
    print("\n🎉 EQUILIBRIUM SOLVER FIXED!")
    print("✅ Both stakeholders profitable")
    print("✅ Meaningful coverage achieved")
    print("✅ Ready for theoretical verification")
else:
    print("\n⚠️  Equilibrium still needs adjustment")
