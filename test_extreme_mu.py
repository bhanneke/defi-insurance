from defi_insurance_core import MarketParameters, InsuranceMarket

# Try extreme mu
params = MarketParameters(
    mu=1000.0,       # Extreme amplification
    theta=0.5,       
    alpha=0.7,       
    u_target=0.2,    
    r_pool=0.10,     
    r_market=0.05,   
    rho=0.03         
)

market = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c = 20_000_000    
market.c_lp = 40_000_000   

state = market.get_market_state()
print("Extreme mu test:")
print(f"Coverage: ${state['coverage']:,.0f} ({state['coverage']/state['tvl']:.1%} of TVL)")
print(f"Utilization: {state['utilization']:.3f}")
print(f"Revenue Share: {state['revenue_share']:.3f}")

# Test profitability
p_hack = 0.1
expected_lgh = 0.1
protocol_profit = market.protocol_profit(market.c_c, market.c_lp, market.c_spec, p_hack, expected_lgh, state['revenue_share'])
lp_profit = market.lp_profit(market.c_c, market.c_lp, market.c_spec, p_hack, expected_lgh, state['revenue_share'])

print(f"Protocol Profit: ${protocol_profit:,.0f}")
print(f"LP Profit: ${lp_profit:,.0f}")
print(f"Both profitable: {protocol_profit > 0 and lp_profit > 0}")
