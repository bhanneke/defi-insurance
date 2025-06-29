#!/usr/bin/env python3
"""
DeFi Insurance Protocol - Full Analysis Script
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("\n" + "="*60)
print("   DeFi Insurance Protocol - Full Analysis")
print("="*60 + "\n")

# Get configuration
initial_capital = float(os.getenv('INITIAL_CAPITAL', 10000000))
num_simulations = int(os.getenv('NUM_SIMULATIONS', 1000))
num_wallets = int(os.getenv('INITIAL_WALLETS', 1000))
test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'

if test_mode:
    print("🧪 Running in TEST MODE (reduced dataset)")
    num_simulations = 10
    num_wallets = 100

print(f"Configuration:")
print(f"  Initial Capital: ${initial_capital:,.0f}")
print(f"  Wallets: {num_wallets}")
print(f"  Simulations: {num_simulations}")
print()

try:
    # Import modules
    from defi_insurance_sim import InsuranceProtocol, MonteCarloSimulator, run_example_simulation
    from defi_insurance_advanced import demonstrate_advanced_features
    from defi_insurance_visualization import run_complete_analysis
    
    # Run analysis
    print("Starting analysis...")
    
    # Option 1: Use the pre-built complete analysis
    if hasattr(run_complete_analysis, '__call__'):
        results = run_complete_analysis()
    else:
        # Option 2: Build our own analysis
        protocol = InsuranceProtocol(initial_capital, 0.02)
        protocol.generate_wallet_population(num_wallets)
        
        simulator = MonteCarloSimulator(protocol, num_simulations)
        results = simulator.run_simulation(time_horizon_years=3)
        analysis = simulator.analyze_results()
        
        print(f"\nResults:")
        print(f"  Profitability: {analysis['profitability_probability']:.1%}")
        print(f"  Expected ROI: {analysis['expected_roi']:.1%}")
        print(f"  Self-Paying Ratio: {analysis['avg_self_paying_ratio']:.1%}")
    
    print("\n✅ Analysis complete!")
    print("\nCheck the following directories:")
    print("  reports/ - Analysis reports")
    print("  visualizations/ - Charts")
    
except Exception as e:
    print(f"\n❌ Error during analysis: {e}")
    import traceback
    traceback.print_exc()
