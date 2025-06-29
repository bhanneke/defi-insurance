#!/usr/bin/env python3
"""
DeFi Insurance Protocol - Main Script
Simple test to verify installation
"""

import os
import sys

print("\n🚀 DeFi Insurance Protocol")
print("="*40)

# Check if simulation files exist
required_files = [
    "defi_insurance_sim.py",
    "defi_insurance_advanced.py",
    "defi_insurance_visualization.py",
    "defi_insurance_production.py"
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print("\n⚠️  Missing required simulation files:")
    for file in missing_files:
        print(f"   - {file}")
    print("\nPlease ensure all simulation files are in the current directory.")
    print("\nRequired files:")
    print("1. defi_insurance_sim.py")
    print("2. defi_insurance_advanced.py")
    print("3. defi_insurance_visualization.py")
    print("4. defi_insurance_production.py")
    sys.exit(1)

# If all files exist, try to import and run
try:
    print("\n✅ All files found. Importing modules...")
    from defi_insurance_sim import InsuranceProtocol, MonteCarloSimulator
    
    print("✅ Imports successful!")
    print("\nRunning quick test...")
    
    # Quick test
    protocol = InsuranceProtocol(1000000, 0.02)
    protocol.generate_wallet_population(10)
    print(f"✅ Created test protocol with {len(protocol.wallets)} wallets")
    
    print("\n🎉 Setup verified! You can now run the full analysis.")
    print("\nTo run complete analysis, use:")
    print("   python main_analysis.py")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nMake sure all dependencies are installed:")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"\n❌ Error: {e}")
