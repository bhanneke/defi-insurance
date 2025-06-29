# Force complete reload
import sys
import importlib
import os

# Remove from cache
if 'defi_insurance_core' in sys.modules:
    del sys.modules['defi_insurance_core']

# Import fresh
from defi_insurance_core import InsuranceMarket, MarketParameters

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

print("🧪 Testing LP Risk Compensation Fix")
print("=" * 40)

c_c = 4_545_455
c_spec = 3_000_000
c_lp_test = 20_000_000

coverage = market.coverage_function(c_c, market.tvl)
u = coverage / c_lp_test
gamma = market.revenue_share_function(u, market.calculate_weighted_risk_price())

p_hack = 0.1
expected_lgh = 0.1

# Test if the function accepts risk_compensation parameter
try:
    lp_profit_old = market.lp_profit(c_c, c_lp_test, c_spec, p_hack, expected_lgh, gamma, risk_compensation=0.0)
    lp_profit_new = market.lp_profit(c_c, c_lp_test, c_spec, p_hack, expected_lgh, gamma, risk_compensation=2.0)
    
    print(f"✅ LP function updated successfully!")
    print(f"LP profit with no risk compensation: ${lp_profit_old:,.0f}")
    print(f"LP profit with 2x risk compensation: ${lp_profit_new:,.0f}")
    print(f"Risk compensation value: ${lp_profit_new - lp_profit_old:,.0f}")
    
    # Test across different capital levels with risk compensation
    print(f"\nLP profits with proper risk compensation:")
    c_lp_range = [5_000_000, 10_000_000, 20_000_000, 40_000_000, 60_000_000]
    
    for c_lp in c_lp_range:
        coverage = market.coverage_function(c_c, market.tvl)
        u = coverage / c_lp
        gamma = market.revenue_share_function(u, market.calculate_weighted_risk_price())
        lp_profit = market.lp_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma, risk_compensation=2.0)
        
        print(f"  C_LP=${c_lp/1e6:.0f}M -> Profit=${lp_profit:,.0f}")
        
except TypeError as e:
    print(f"❌ LP function update failed: {e}")
    print("The risk_compensation parameter was not added successfully")
