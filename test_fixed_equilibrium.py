# Force reload the module to get the updated protocol_profit function
import sys
import importlib

# Remove from cache and reload
if 'defi_insurance_core' in sys.modules:
    del sys.modules['defi_insurance_core']

from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

print("🔧 Testing Fixed Equilibrium Solver with Risk Aversion")
print("=" * 55)

test_c_lp = 40_000_000
test_c_spec = 3_000_000

# Test the updated protocol_profit function with risk_aversion parameter
c_c_range = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000]

for c_c in c_c_range:
    coverage = market.coverage_function(c_c, market.tvl)
    u = coverage / test_c_lp if test_c_lp > 0 else float('inf')
    gamma = market.revenue_share_function(u, market.calculate_weighted_risk_price())
    
    p_hack = 0.1
    expected_lgh = 0.1
    
    # Test both old (risk_aversion=0) and new (risk_aversion=2.0) profit
    try:
        profit_no_risk = market.protocol_profit(c_c, test_c_lp, test_c_spec, p_hack, expected_lgh, gamma, risk_aversion=0.0)
        profit_with_risk = market.protocol_profit(c_c, test_c_lp, test_c_spec, p_hack, expected_lgh, gamma, risk_aversion=2.0)
        
        print(f"  C_C=${c_c/1e6:.0f}M -> No Risk: ${profit_no_risk:,.0f}, With Risk: ${profit_with_risk:,.0f}")
        
    except TypeError as e:
        print(f"  Error: {e}")
        print("  The protocol_profit function update failed")
        break

# Now test the equilibrium solver
print(f"\nTesting equilibrium solver:")
solver = EquilibriumSolver(market)
optimal_c_c = solver.protocol_best_response(test_c_lp, test_c_spec)
print(f"Best response returns: C_C=${optimal_c_c:,.0f}")

# Test full equilibrium
print(f"\nTesting full equilibrium:")
c_c, c_lp, c_spec = solver.find_equilibrium(max_iterations=20)
print(f"Full equilibrium: C_C=${c_c:,.0f}, C_LP=${c_lp:,.0f}, C_spec=${c_spec:,.0f}")
