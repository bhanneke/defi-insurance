from defi_insurance_core import InsuranceMarket, MarketParameters

params = MarketParameters(mu=1.5, theta=0.7)
market = InsuranceMarket(params)
market.tvl = 100_000_000

# Set reasonable values
market.c_c = 15_000_000   
market.c_lp = 60_000_000  
market.c_spec = 3_000_000 

# Test basic functionality
state = market.get_market_state()
print("Market Test:")
print(f"Coverage: ${state['coverage']:,.0f}")
print(f"Utilization: {state['utilization']:.4f}")
print(f"Revenue Share: {state['revenue_share']:.3f}")

# Test profit calculations
p_hack = 0.1
expected_lgh = 0.1
protocol_profit = market.protocol_profit(market.c_c, market.c_lp, market.c_spec, p_hack, expected_lgh, state['revenue_share'])
lp_profit = market.lp_profit(market.c_c, market.c_lp, market.c_spec, p_hack, expected_lgh, state['revenue_share'])

print(f"Protocol Profit: ${protocol_profit:,.0f}")
print(f"LP Profit: ${lp_profit:,.0f}")
