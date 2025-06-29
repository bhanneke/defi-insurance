from defi_insurance_core import InsuranceMarket, MarketParameters

# Increase alpha to give LPs more revenue share
params = MarketParameters(mu=1.5, theta=0.7, alpha=0.9, u_target=0.8)
market = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c = 15_000_000   
market.c_lp = 60_000_000  
market.c_spec = 3_000_000 

state = market.get_market_state()
print("Higher LP Share Test:")
print(f"Revenue Share: {state['revenue_share']:.3f}")

p_hack = 0.1
expected_lgh = 0.1
protocol_profit = market.protocol_profit(market.c_c, market.c_lp, market.c_spec, p_hack, expected_lgh, state['revenue_share'])
lp_profit = market.lp_profit(market.c_c, market.c_lp, market.c_spec, p_hack, expected_lgh, state['revenue_share'])

print(f"Protocol Profit: ${protocol_profit:,.0f}")
print(f"LP Profit: ${lp_profit:,.0f}")

if protocol_profit > 0 and lp_profit > 0:
    print("✅ Both stakeholders profitable!")
else:
    print("❌ Still not balanced")
