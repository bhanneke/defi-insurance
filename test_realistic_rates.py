from defi_insurance_core import InsuranceMarket, MarketParameters

print("🧪 Testing with Realistic Equal Rates")
print("=" * 40)

# Test different rate scenarios
scenarios = [
    {"name": "Current (Subsidized)", "r_market": 0.05, "r_pool": 0.10},
    {"name": "Equal Rates", "r_market": 0.05, "r_pool": 0.05},
    {"name": "Pool Penalty", "r_market": 0.05, "r_pool": 0.04},  # Pool less efficient
]

for scenario in scenarios:
    print(f"\n📊 {scenario['name']}:")
    print(f"   Market Rate: {scenario['r_market']:.1%}")
    print(f"   Pool Yield: {scenario['r_pool']:.1%}")
    
    params = MarketParameters(
        mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2,
        r_market=scenario['r_market'], 
        r_pool=scenario['r_pool'], 
        rho=0.03
    )
    
    market = InsuranceMarket(params)
    market.tvl = 100_000_000
    market.c_c = 252_526
    market.c_lp = 1_011_096
    market.c_spec = 3_000_000
    
    state = market.get_market_state()
    
    p_hack, expected_lgh = 0.1, 0.1
    protocol_profit = market.protocol_profit(
        market.c_c, market.c_lp, market.c_spec, 
        p_hack, expected_lgh, state['revenue_share'],
        risk_aversion=20.0
    )
    lp_profit = market.lp_profit(
        market.c_c, market.c_lp, market.c_spec,
        p_hack, expected_lgh, state['revenue_share'],
        risk_compensation=2.0
    )
    
    print(f"   Protocol Profit: ${protocol_profit:,.0f}")
    print(f"   LP Profit: ${lp_profit:,.0f}")
    print(f"   Both Profitable: {protocol_profit > 0 and lp_profit > 0}")
    
    # Calculate the subsidy effect
    if scenario['name'] == 'Current (Subsidized)':
        subsidy_value = (scenario['r_pool'] - scenario['r_market']) * (market.c_c + market.c_lp + market.c_spec)
        print(f"   Hidden Subsidy: ${subsidy_value:,.0f}")

print(f"\n💡 Analysis:")
print(f"If the mechanism only works with subsidized pool yields,")
print(f"then it's not truly economically viable on its own.")
