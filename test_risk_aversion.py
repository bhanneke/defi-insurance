from defi_insurance_core import InsuranceMarket, MarketParameters
import math

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

def protocol_profit_with_risk_aversion(c_c, c_lp, c_spec, risk_aversion=2.0):
    """
    Modified protocol profit with risk aversion utility
    
    Args:
        risk_aversion: Risk aversion coefficient (higher = more risk averse)
    """
    # Original profit calculation
    coverage = market.coverage_function(c_c, market.tvl)
    u = coverage / c_lp if c_lp > 0 else float('inf')
    gamma = market.revenue_share_function(u, market.calculate_weighted_risk_price())
    
    p_hack = 0.1
    expected_lgh = 0.1
    c_premium = market.params.premium_rate * market.tvl
    total_cap = c_c + c_premium + c_lp + c_spec
    
    # Expected insurance payout
    expected_loss = expected_lgh * market.tvl
    insurance_payout = p_hack * min(coverage, expected_loss)
    
    # Protocol's share of pool yield
    yield_share = (1 - gamma) * market.params.r_pool * total_cap
    
    # Opportunity cost
    opportunity_cost = c_c * market.params.r_market
    premium_cost = c_premium
    
    # Basic profit (without risk aversion)
    basic_profit = insurance_payout + yield_share - opportunity_cost - premium_cost
    
    # Risk aversion utility: Value of risk reduction
    # Utility from reducing potential loss exposure
    uninsured_loss_risk = p_hack * expected_loss  # Expected loss without insurance
    insured_loss_risk = p_hack * max(0, expected_loss - coverage)  # Expected loss with insurance
    risk_reduction_value = (uninsured_loss_risk - insured_loss_risk) * risk_aversion
    
    # Total utility = basic profit + risk reduction value
    total_utility = basic_profit + risk_reduction_value
    
    return total_utility, basic_profit, risk_reduction_value

print("🎯 Testing Protocol Profit with Risk Aversion")
print("=" * 50)

test_c_lp = 40_000_000
test_c_spec = 3_000_000

c_c_range = [0, 5_000_000, 10_000_000, 20_000_000, 30_000_000]

for c_c in c_c_range:
    total_utility, basic_profit, risk_value = protocol_profit_with_risk_aversion(c_c, test_c_lp, test_c_spec, risk_aversion=2.0)
    coverage = market.coverage_function(c_c, market.tvl)
    
    print(f"C_C=${c_c/1e6:.0f}M:")
    print(f"  Coverage: ${coverage:,.0f}")
    print(f"  Basic Profit: ${basic_profit:,.0f}")
    print(f"  Risk Reduction Value: ${risk_value:,.0f}")
    print(f"  Total Utility: ${total_utility:,.0f}")
    print()

# Find optimal with different risk aversion levels
print("Optimal collateral for different risk aversion levels:")
for risk_aversion in [0.5, 1.0, 2.0, 5.0]:
    max_utility = -float('inf')
    best_c_c = 0
    
    for c_c in range(0, 50_000_000, 1_000_000):
        utility, _, _ = protocol_profit_with_risk_aversion(c_c, test_c_lp, test_c_spec, risk_aversion)
        if utility > max_utility:
            max_utility = utility
            best_c_c = c_c
    
    print(f"  Risk aversion {risk_aversion}: Optimal C_C=${best_c_c/1e6:.1f}M, Utility=${max_utility:,.0f}")
