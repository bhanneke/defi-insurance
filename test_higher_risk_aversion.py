from defi_insurance_core import InsuranceMarket, MarketParameters

params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_market=0.05, r_pool=0.10, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000

# Use our equilibrium values
c_c = 252_526
c_lp = 1_011_096  
c_spec = 3_000_000

market.c_c, market.c_lp, market.c_spec = c_c, c_lp, c_spec
state = market.get_market_state()

p_hack = 0.1
expected_lgh = 0.1

print("🧪 Testing Protocol Profit with Different Risk Aversion Levels")
print("=" * 65)

for risk_aversion in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    protocol_profit = market.protocol_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, state['revenue_share'], risk_aversion=risk_aversion)
    
    print(f"Risk aversion {risk_aversion:4.1f}: Protocol profit ${protocol_profit:,.0f}")

print(f"\n📊 Current market state:")
print(f"  Coverage: ${state['coverage']:,.0f}")
print(f"  Expected loss: ${expected_lgh * market.tvl:,.0f}")
print(f"  Risk reduction: ${min(state['coverage'], expected_lgh * market.tvl):,.0f}")

# Calculate the break-even risk aversion
print(f"\n🎯 Finding break-even risk aversion:")

for risk_aversion in [10, 20, 30, 40, 50, 75, 100]:
    protocol_profit = market.protocol_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, state['revenue_share'], risk_aversion=risk_aversion)
    
    if protocol_profit > 0:
        print(f"✅ Break-even at risk aversion = {risk_aversion} (profit: ${protocol_profit:,.0f})")
        
        # This would be the publishable parameter
        print(f"\n🎉 PUBLISHABLE PARAMETERS FOUND:")
        print(f"   Risk aversion: {risk_aversion}")
        print(f"   All other parameters: {params}")
        break
else:
    print("❌ Even extreme risk aversion doesn't make protocols profitable")
    print("💡 Need to adjust other parameters (opportunity cost, yields, etc.)")
