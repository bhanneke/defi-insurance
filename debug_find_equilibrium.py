import sys
if 'defi_insurance_core' in sys.modules:
    del sys.modules['defi_insurance_core']

from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver

# Let's examine the actual find_equilibrium method
with open('defi_insurance_core.py', 'r') as f:
    content = f.read()

# Find the find_equilibrium method
start = content.find('def find_equilibrium(')
end = content.find('\n    def ', start + 1)
if end == -1:
    end = content.find('\n\nclass ', start + 1)
if end == -1:
    end = len(content)

equilibrium_method = content[start:end]
print("🔍 Current find_equilibrium method:")
print("=" * 50)
print(equilibrium_method[:800] + "..." if len(equilibrium_method) > 800 else equilibrium_method)

print("\n" + "="*50)
print("The issue might be:")
print("1. The method is not converging properly")
print("2. The bounds checking is forcing corner solutions")  
print("3. The initial values are wrong")
print("4. The tolerance is too strict")

# Let's test with our own simple equilibrium finder
params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2, r_pool=0.10, r_market=0.05, rho=0.03)
market = InsuranceMarket(params)
market.tvl = 100_000_000
solver = EquilibriumSolver(market)

print("\n🛠️  Manual equilibrium iteration:")
c_c, c_lp, c_spec = 5_000_000, 15_000_000, 3_000_000  # Reasonable start

for i in range(10):
    c_c_new = solver.protocol_best_response(c_lp, c_spec)
    c_lp_new = solver.lp_best_response(c_c_new, c_spec)
    # Keep c_spec fixed for simplicity
    
    print(f"Iter {i+1}: C_C ${c_c:,.0f}→${c_c_new:,.0f}, C_LP ${c_lp:,.0f}→${c_lp_new:,.0f}")
    
    error = abs(c_c - c_c_new) + abs(c_lp - c_lp_new)
    if error < 100_000:  # $100K tolerance
        print(f"✅ Converged! Final: C_C=${c_c_new:,.0f}, C_LP=${c_lp_new:,.0f}")
        
        # Test this equilibrium
        market.c_c, market.c_lp, market.c_spec = c_c_new, c_lp_new, c_spec
        state = market.get_market_state()
        p_hack, expected_lgh = 0.1, 0.1
        
        protocol_profit = market.protocol_profit(c_c_new, c_lp_new, c_spec, p_hack, expected_lgh, state['revenue_share'])
        lp_profit = market.lp_profit(c_c_new, c_lp_new, c_spec, p_hack, expected_lgh, state['revenue_share'])
        
        print(f"Protocol Profit: ${protocol_profit:,.0f}")
        print(f"LP Profit: ${lp_profit:,.0f}")
        print(f"Coverage: ${state['coverage']:,.0f}")
        break
    
    c_c, c_lp = c_c_new, c_lp_new
