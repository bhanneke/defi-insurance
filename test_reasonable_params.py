from defi_insurance_core import MarketParameters, InsuranceMarket

# Hand-picked reasonable parameters
params = MarketParameters(
    mu=10.0,         # Higher amplification for meaningful coverage
    theta=0.5,       # Moderate concavity
    alpha=0.7,       # LP gets 70% weight in revenue share
    u_target=0.2,    # Target 20% utilization (realistic)
    beta=1.0,        # Linear utilization effect
    r_pool=0.10,     # 10% pool yield
    r_market=0.05,   # 5% market rate
    rho=0.03         # 3% risk premium
)

market = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c = 20_000_000    # 20% of TVL
market.c_lp = 40_000_000   # 40% of TVL

state = market.get_market_state()
print("Hand-picked parameters:")
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
