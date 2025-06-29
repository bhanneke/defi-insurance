from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

print("🔍 Debugging Equilibrium Solver")
print("=" * 40)

# Test the best response functions individually
solver = EquilibriumSolver(market)

# Test protocol best response with reasonable LP values
test_c_lp = 40_000_000
test_c_spec = 3_000_000

print(f"Testing protocol best response with C_LP=${test_c_lp:,}, C_spec=${test_c_spec:,}")

# Check what the protocol best response function does
c_c_range = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000]

for c_c in c_c_range:
    # Calculate profit for this collateral amount
    coverage = market.coverage_function(c_c, market.tvl)
    u = coverage / test_c_lp if test_c_lp > 0 else float('inf')
    gamma = market.revenue_share_function(u, market.calculate_weighted_risk_price())
    
    p_hack = 0.1
    expected_lgh = 0.1
    profit = market.protocol_profit(c_c, test_c_lp, test_c_spec, p_hack, expected_lgh, gamma)
    
    print(f"  C_C=${c_c/1e6:.0f}M -> Profit=${profit:,.0f}, Coverage=${coverage:,.0f}, Util={u:.3f}, γ={gamma:.3f}")

print(f"\nNow testing the best response function:")
optimal_c_c = solver.protocol_best_response(test_c_lp, test_c_spec)
print(f"Best response returns: C_C=${optimal_c_c:,.0f}")
