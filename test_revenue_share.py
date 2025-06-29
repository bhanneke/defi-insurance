from defi_insurance_core import InsuranceMarket, MarketParameters

params = MarketParameters(mu=1.5, theta=0.7, alpha=0.8, beta=1.5)
market = InsuranceMarket(params)

# Test different utilization levels
for c_lp in [5_000_000, 10_000_000, 20_000_000, 40_000_000]:
    market.tvl = 100_000_000
    market.c_c = 15_000_000   
    market.c_lp = c_lp
    
    state = market.get_market_state()
    print(f"LP Capital: ${c_lp/1e6:.0f}M, Utilization: {state['utilization']:.3f}, Revenue Share: {state['revenue_share']:.3f}")

print("\nThe revenue share formula penalizes low utilization!")
print("We need higher utilization for LPs to get decent revenue share.")
