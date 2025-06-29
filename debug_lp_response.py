import sys
if 'defi_insurance_core' in sys.modules:
    del sys.modules['defi_insurance_core']

from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

print("🔍 Debugging LP Best Response")
print("=" * 35)

# Test LP profits across different capital levels
c_c = 4_545_455  # From protocol best response
c_spec = 3_000_000

c_lp_range = [1_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000, 60_000_000]

for c_lp in c_lp_range:
    coverage = market.coverage_function(c_c, market.tvl)
    u = coverage / c_lp if c_lp > 0 else float('inf')
    gamma = market.revenue_share_function(u, market.calculate_weighted_risk_price())
    
    p_hack = 0.1
    expected_lgh = 0.1
    lp_profit = market.lp_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma)
    
    print(f"  C_LP=${c_lp/1e6:.1f}M -> Profit=${lp_profit:,.0f}, Util={u:.3f}, γ={gamma:.3f}")

# Test the LP best response function
solver = EquilibriumSolver(market)
optimal_c_lp = solver.lp_best_response(c_c, c_spec)
print(f"\nLP best response: ${optimal_c_lp:,.0f}")

print(f"\nThe LP best response function is probably finding the minimum capital")
print(f"because LP profits decrease as LP capital increases!")
