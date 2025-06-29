#!/usr/bin/env python3
"""
DeFi Insurance Protocol - Simple Setup Script
This is a simplified version that should work immediately
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Colors for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Print colored message"""
    print(f"{color}{message}{Colors.RESET}")

def print_banner():
    """Print welcome banner"""
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("   DeFi Insurance Protocol - Setup Script", Colors.CYAN)
    print_colored("="*60 + "\n", Colors.CYAN)

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored(f"❌ Python 3.8+ required, but {version.major}.{version.minor} found", Colors.RED)
        sys.exit(1)
 2   print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} detected", Colors.GREEN)

def create_requirements_file():
    """Create requirements.txt"""
    requirements = """# DeFi Insurance Protocol Requirements
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
plotly>=5.3.0
scikit-learn>=0.24.0
scikit-optimize>=0.9.0
joblib>=1.0.0
schedule>=1.1.0
python-dotenv>=0.19.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print_colored("✅ Created requirements.txt", Colors.GREEN)

def create_env_file():
    """Create .env file"""
    env_content = """# DeFi Insurance Protocol Configuration

# Basic Settings
INITIAL_CAPITAL=10000000
BASE_PREMIUM_RATE=0.02
NUM_SIMULATIONS=1000
TIME_HORIZON_YEARS=3
INITIAL_WALLETS=1000

# Token Parameters
TOKEN_DISTRIBUTION_RATE=0.05
TOKEN_STAKING_APY=0.08
TOKEN_REVENUE_SHARING_RATIO=0.30

# For quick testing, uncomment these:
# TEST_MODE=true
# NUM_SIMULATIONS=10
# INITIAL_WALLETS=100
"""
    
    if os.path.exists(".env"):
        response = input(".env file exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            return
    
    with open(".env", "w") as f:
        f.write(env_content)
    print_colored("✅ Created .env file", Colors.GREEN)

def create_main_script():
    """Create main.py"""
    main_content = '''#!/usr/bin/env python3
"""
DeFi Insurance Protocol - Main Script
Simple test to verify installation
"""

import os
import sys

print("\\n🚀 DeFi Insurance Protocol")
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
    print("\\n⚠️  Missing required simulation files:")
    for file in missing_files:
        print(f"   - {file}")
    print("\\nPlease ensure all simulation files are in the current directory.")
    print("\\nRequired files:")
    print("1. defi_insurance_sim.py")
    print("2. defi_insurance_advanced.py")
    print("3. defi_insurance_visualization.py")
    print("4. defi_insurance_production.py")
    sys.exit(1)

# If all files exist, try to import and run
try:
    print("\\n✅ All files found. Importing modules...")
    from defi_insurance_sim import InsuranceProtocol, MonteCarloSimulator
    
    print("✅ Imports successful!")
    print("\\nRunning quick test...")
    
    # Quick test
    protocol = InsuranceProtocol(1000000, 0.02)
    protocol.generate_wallet_population(10)
    print(f"✅ Created test protocol with {len(protocol.wallets)} wallets")
    
    print("\\n🎉 Setup verified! You can now run the full analysis.")
    print("\\nTo run complete analysis, use:")
    print("   python main_analysis.py")
    
except ImportError as e:
    print(f"\\n❌ Import error: {e}")
    print("\\nMake sure all dependencies are installed:")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"\\n❌ Error: {e}")
'''
    
    with open("main.py", "w") as f:
        f.write(main_content)
    print_colored("✅ Created main.py", Colors.GREEN)

def create_analysis_script():
    """Create the full analysis script"""
    analysis_content = '''#!/usr/bin/env python3
"""
DeFi Insurance Protocol - Full Analysis Script
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("\\n" + "="*60)
print("   DeFi Insurance Protocol - Full Analysis")
print("="*60 + "\\n")

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
        
        print(f"\\nResults:")
        print(f"  Profitability: {analysis['profitability_probability']:.1%}")
        print(f"  Expected ROI: {analysis['expected_roi']:.1%}")
        print(f"  Self-Paying Ratio: {analysis['avg_self_paying_ratio']:.1%}")
    
    print("\\n✅ Analysis complete!")
    print("\\nCheck the following directories:")
    print("  reports/ - Analysis reports")
    print("  visualizations/ - Charts")
    
except Exception as e:
    print(f"\\n❌ Error during analysis: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open("main_analysis.py", "w") as f:
        f.write(analysis_content)
    print_colored("✅ Created main_analysis.py", Colors.GREEN)

def setup_virtual_environment():
    """Create and setup virtual environment"""
    venv_path = "venv"
    
    # Check if venv exists
    if os.path.exists(venv_path):
        response = input("\nVirtual environment exists. Recreate? (y/n): ")
        if response.lower() != 'y':
            return True
        
        # Remove existing venv
        import shutil
        shutil.rmtree(venv_path)
    
    print_colored("\nCreating virtual environment...", Colors.BLUE)
    
    try:
        # Create venv
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print_colored("✅ Virtual environment created", Colors.GREEN)
        
        # Determine pip path
        if platform.system() == "Windows":
            pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
            python_path = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            pip_path = os.path.join(venv_path, "bin", "pip")
            python_path = os.path.join(venv_path, "bin", "python")
        
        # Upgrade pip
        print_colored("\nUpgrading pip...", Colors.BLUE)
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        if os.path.exists("requirements.txt"):
            print_colored("\nInstalling requirements...", Colors.BLUE)
            subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
            print_colored("✅ All dependencies installed", Colors.GREEN)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to setup virtual environment: {e}", Colors.RED)
        return False

def create_directories():
    """Create project directories"""
    dirs = ["logs", "reports", "visualizations", "data", "cache"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print_colored("✅ Created project directories", Colors.GREEN)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Create necessary files
    print_colored("\n📁 Creating configuration files...", Colors.BLUE)
    create_requirements_file()
    create_env_file()
    create_main_script()
    create_analysis_script()
    
    # Create directories
    create_directories()
    
    # Setup virtual environment
    if setup_virtual_environment():
        print_colored("\n✅ Setup completed successfully!", Colors.GREEN)
        
        print_colored("\n📋 Next steps:", Colors.YELLOW)
        print("1. Copy the 4 simulation Python files to this directory:")
        print("   - defi_insurance_sim.py")
        print("   - defi_insurance_advanced.py")
        print("   - defi_insurance_visualization.py")
        print("   - defi_insurance_production.py")
        print("\n2. Activate the virtual environment:")
        if platform.system() == "Windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("\n3. Test the setup:")
        print("   python main.py")
        print("\n4. Run full analysis:")
        print("   python main_analysis.py")
        
        print_colored("\n💡 Tip: Edit .env file to customize parameters", Colors.CYAN)
    else:
        print_colored("\n❌ Setup failed. Please check the errors above.", Colors.RED)

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir("/Users/hanneke/Documents/Projects/Web3_Insurance/code_260626")
    main()