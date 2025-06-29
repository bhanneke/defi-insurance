from defi_insurance_core import InsuranceMarket, MarketParameters

# Try more conservative parameters
params = MarketParameters(
    mu=1.5,      # Lower amplification
    theta=0.7,   # Higher concavity to limit growth
    alpha=0.6,
    u_target=0.8
)

market = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c = 10_000_000
market.c_lp = 50_000_000
market.c_spec = 2_000_000

state = market.get_market_state()
print(f'Stable test - Coverage: ${state["coverage"]:,.0f}')
print(f'Stable test - Utilization: {state["utilization"]:.2f}')
print(f'Coverage ratio: {state["coverage"]/state["tvl"]:.1%}')
