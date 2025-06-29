## work in progress ##
# Decentralized Finance: A Market Mechanism for Cybersecurity Risk Insurance

This repository contains the complete implementation and replication code for the academic paper **"Decentralized Finance: A Market Mechanism for Cybersecurity Risk Insurance"** by Björn Hanneke (Goethe University Frankfurt).

## 📄 Paper Abstract

The rapid growth of Decentralized Finance has introduced significant cybersecurity risks, with protocol vulnerabilities leading to substantial financial losses exceeding $11 billion. This research proposes a novel market-based model for decentralized cybersecurity risk insurance, leveraging Automated Market Makers and collateral-based coverage. The mechanism enables protocols to access capital-efficient coverage while risk is priced continuously through trading of "Loss Given Hack" (LGH) tokens.

## 🎯 Research Contributions

- **Market-based risk pricing**: Eliminates reliance on actuarial assumptions through continuous AMM-based price discovery
- **Incentive alignment**: Protocol collateral requirements mitigate moral hazard while dynamic yield adjustments sustain liquidity provision
- **Capital efficiency**: Separation of risk bearing (LPs) from risk pricing (speculators) enables optimal capital allocation
- **Formal analysis**: Mathematical proofs of equilibrium existence, capital self-stabilization, and sustainable coverage bounds

## 📁 Repository Structure

```
├── defi_insurance_core.py      # Core mechanism implementation
├── simulation_module.py        # Monte Carlo simulations and scenario analysis
├── theoretical_proofs.py       # Mathematical proofs and analytical validation
├── parameter_optimization.py   # Parameter calibration and sensitivity analysis
├── main_runner.py             # Main execution script for all analyses
├── results/                   # Generated results and outputs
├── final_results/            # Final paper results and figures
├── requirements.txt          # Python dependencies
├── .env.template            # Environment configuration template
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhanneke/defi-insurance.git
   cd defi-insurance
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment (optional):**
   ```bash
   cp .env.template .env
   # Edit .env with your preferred settings if needed
   ```

### Running the Analysis

#### Quick Verification (30 seconds)
Test that the mechanism works:
```bash
python -c "
from defi_insurance_core import InsuranceMarket, MarketParameters
params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2)
market = InsuranceMarket(params)
market.tvl = 100_000_000
market.c_c, market.c_lp, market.c_spec = 252_526, 1_011_096, 3_000_000
coverage = market.coverage_function(market.c_c, market.tvl)
print(f'Coverage: \${coverage:,.0f} ({coverage/market.tvl:.1%} of TVL)')
print('✅ Mechanism works!')
"
```

#### Complete Replication
To reproduce all paper results:
```bash
python main_runner.py
```

#### Individual Components

**1. Core Mechanism Analysis:**
```python
from defi_insurance_core import InsuranceMarket, MarketParameters

# Initialize the insurance model
params = MarketParameters(mu=1000.0, theta=0.5, alpha=0.7, u_target=0.2)
market = InsuranceMarket(params)

# Set market conditions
market.tvl = 100_000_000  # $100M TVL
market.c_c = 5_000_000    # $5M protocol collateral
market.c_lp = 10_000_000  # $10M LP capital

# Calculate coverage and market state
coverage = market.coverage_function(market.c_c, market.tvl)
market_state = market.get_market_state()
print(f"Maximum coverage: ${coverage:,.0f}")
print(f"Utilization: {market.utilization():.2%}")
```

**2. Monte Carlo Simulations:**
```python
from simulation_module import run_monte_carlo_simulation

# Run scenario analysis
results = run_monte_carlo_simulation(
    n_simulations=10000,
    time_horizon=365,  # days
    protocols=50
)
```

**3. Theoretical Validation:**
```python
from theoretical_proofs import verify_equilibrium_existence, test_incentive_compatibility

# Verify theoretical propositions
equilibrium_exists = verify_equilibrium_existence()
incentives_aligned = test_incentive_compatibility()
```

**4. Parameter Optimization:**
```python
from parameter_optimization import optimize_mechanism_parameters

# Find optimal parameter calibration
optimal_params = optimize_mechanism_parameters()
```

## 📊 Key Results Reproduction

The code reproduces the following key findings from the paper:

### 1. **Equilibrium Analysis** (`theoretical_proofs.py`)
- **Theorem 1**: Existence of three-party Nash equilibrium
- **Proposition 1**: Truthful risk assessment in LGH markets
- **Theorem 2**: LP dynamics and participation bounds

### 2. **Mechanism Properties** (`defi_insurance_core.py`)
- Coverage function with diminishing marginal returns
- Dynamic revenue sharing based on utilization and risk pricing
- Automated market maker implementation for LGH tokens

### 3. **Economic Simulations** (`simulation_module.py`)
- Protocol behavior under different collateral scenarios
- LP capital flows and yield optimization
- Market stability under varying risk conditions

### 4. **Parameter Sensitivity** (`parameter_optimization.py`)
- Coverage amplification factor (μ) calibration
- Revenue share function parameters (α, β, δ)
- Utilization target optimization

## 🔧 Key Model Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `μ` | Coverage amplification factor | 2.0 |
| `θ` | Coverage concavity parameter | 0.7 |
| `α` | Weight on utilization in revenue share | 0.6 |
| `β, δ` | Convexity parameters | 1.5, 2.0 |
| `U_target` | Target utilization level | 0.8 |

## 📈 Expected Outputs

Running the complete analysis generates:

1. **Equilibrium Solutions**: Nash equilibrium points for all stakeholders
2. **Simulation Results**: Monte Carlo analysis of mechanism performance
3. **Theoretical Validation**: Verification of formal propositions
4. **Parameter Sensitivity**: Robustness analysis across parameter ranges
5. **Performance Metrics**: Capital efficiency, coverage ratios, and yield distributions

Results are saved in the `results/` and `final_results/` directories with timestamps and parameter configurations.

## 🎛️ Configuration Options

Modify simulation parameters in `main_runner.py`:

```python
# Simulation settings
N_SIMULATIONS = 10000
TIME_HORIZON = 365  # days
N_PROTOCOLS = 50
HACK_PROBABILITY = 0.05  # annual probability

# Market parameters
INITIAL_TVL = 1e9  # $1B
RISK_FREE_RATE = 0.03
LP_RISK_PREMIUM = 0.02
```

## 📋 Paper Replication Checklist

- [ ] **Figure 1**: Market mechanism diagram (conceptual - see paper)
- [ ] **Theorem 1**: Nash equilibrium existence ✓ (`theoretical_proofs.py`)
- [ ] **Proposition 1**: Truthful risk assessment ✓ (`theoretical_proofs.py`)
- [ ] **Theorem 2**: LP dynamics and bounds ✓ (`theoretical_proofs.py`)
- [ ] **Proposition 2**: Solvency bounds ✓ (`theoretical_proofs.py`)
- [ ] **Equation (1-10)**: All pricing formulas ✓ (`defi_insurance_core.py`)
- [ ] **Monte Carlo Results**: Mechanism performance ✓ (`simulation_module.py`)
- [ ] **Parameter Sensitivity**: Robustness analysis ✓ (`parameter_optimization.py`)

## 🔬 Research Extensions

This implementation enables several research extensions:

1. **Multi-protocol analysis**: Extend to interconnected protocol networks
2. **Dynamic parameters**: Time-varying risk and market conditions
3. **Alternative AMM curves**: Different bonding curve implementations
4. **Regulatory scenarios**: Compliance and institutional participation models

## 📚 Citation

If you use this code in your research, please cite:

```
@article{hanneke2025defi,
  title={Decentralized Finance: A Market Mechanism for Cybersecurity Risk Insurance},
  author={Hanneke, Bj{\"o}rn},
  journal={[Journal Name]},
  year={2025},
  institution={Goethe University Frankfurt}
}
```

## 🤝 Contributing

This repository primarily serves academic replication purposes. For questions about the methodology or implementation:

1. Check the paper for theoretical foundations
2. Review code comments for implementation details
3. Open an issue for specific technical questions

## 📄 License

This code is provided for academic research and replication purposes. Please respect academic integrity guidelines and cite appropriately.
