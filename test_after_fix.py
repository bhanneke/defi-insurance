# Test after manually editing the file
import importlib
import sys

# Force reload the module
if 'defi_insurance_core' in sys.modules:
    importlib.reload(sys.modules['defi_insurance_core'])

from defi_insurance_core import InsuranceMarket, MarketParameters

params = MarketParameters(mu=1.5, themarket = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c = state = market.get_market_state()
print(f'After fix - Coverage: ${state["coverage"]:,.0f}')
print(f'After fix - Utilization: {state["utilization"]:.2f}')
