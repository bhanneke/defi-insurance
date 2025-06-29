import sys
if 'defi_insurance_core' in sys.modules:
    del sys.modules['defi_insurance_core']

from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

solver = EquilibriumSolver(market)

print("🔍 Debugging Full Equilibrium Process")
print("=" * 40)

# Start with reasonable initial values
c_c = 10_000_000
c_lp = 40_000_000  
c_spec = 3_000_000

print(f"Starting values: C_C=${c_c:,}, C_LP=${c_lp:,}, C_spec=${c_spec:,}")

# Manual iteration to see what's happening
for iteration in range(5):
    c_c_old, c_lp_old, c_spec_old = c_c, c_lp, c_spec
    
    # Get best responses
    c_c_new = solver.protocol_best_response(c_lp, c_spec)
    c_lp_new = solver.lp_best_response(c_c_new, c_spec)
    c_spec_new = solver._speculator_best_response(c_c_new, c_lp_new)
    
    print(f"\nIteration {iteration + 1}:")
    print(f"  Protocol BR: C_C=${c_c_new:,.0f} (was ${c_c:,.0f})")
    print(f"  LP BR: C_LP=${c_lp_new:,.0f} (was ${c_lp:,.0f})")
    print(f"  Spec BR: C_spec=${c_spec_new:,.0f} (was ${c_spec:,.0f})")
    
    # Update values
    c_c, c_lp, c_spec = c_c_new, c_lp_new, c_spec_new
    
    # Check convergence
    error = abs(c_c - c_c_old) + abs(c_lp - c_lp_old) + abs(c_spec - c_spec_old)
    print(f"  Convergence error: {error:,.0f}")
    
    if error < 1000:
        print(f"  ✅ Converged!")
        break

print(f"\nFinal equilibrium: C_C=${c_c:,.0f}, C_LP=${c_lp:,.0f}, C_spec=${c_spec:,.0f}")

# Test if this equilibrium makes economic sense
market.c_c, market.c_lp, market.c_spec = c_c, c_lp, c_spec
state = market.get_market_state()
print(f"\nMarket state:")
print(f"  Coverage: ${state['coverage']:,.0f}")
print(f"  Utilization: {state['utilization']:.4f}")
print(f"  Revenue Share: {state['revenue_share']:.3f}")
