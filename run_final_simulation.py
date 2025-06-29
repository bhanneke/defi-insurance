from simulation_module import MarketSimulator, SimulationScenario
from defi_insurance_core import MarketParameters

# Use the working parameters
params = MarketParameters(
    mu=1000.0,       
    theta=0.5,       
    alpha=0.7,       
    u_target=0.2,    
    r_pool=0.10,     
    r_market=0.05,   
    rho=0.03         
)

simulator = MarketSimulator(params)

scenario = SimulationScenario(
    name='Working Parameters',
    tvl_range=(80_000_000, 120_000_000),
    hack_probability_range=(0.05, 0.20),
    lgh_severity_range=(0.05, 0.30),
    num_simulations=200,  # Reasonable number
    duration_days=365
)

print("🚀 Running DeFi Insurance Simulation with Working Parameters...")
print(f"Parameters: μ={params.mu}, θ={params.theta}, α={params.alpha}")

results = simulator.run_monte_carlo(scenario)
analysis = simulator.analyze_results(results)

print("\n🎯 SIMULATION RESULTS:")
print(f"Protocol Profitability: {analysis['profitability']['protocol_profit_positive_pct']:.1f}%")
print(f"LP Profitability: {analysis['profitability']['lp_profit_positive_pct']:.1f}%")
print(f"Insolvency Risk: {analysis['risk_metrics']['insolvency_probability']:.1f}%")
print(f"Average Loss Ratio: {analysis['summary_stats']['avg_loss_ratio']:.2f}")
print(f"Average Hack Frequency: {analysis['summary_stats']['avg_hack_frequency']:.3f} per year")
