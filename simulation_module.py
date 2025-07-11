"""
Monte Carlo Simulation for DeFi Insurance Market
Simulates market dynamics, stress testing, and scenario analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from defi_insurance_core import InsuranceMarket, MarketParameters, EquilibriumSolver


@dataclass
class SimulationScenario:
    """Configuration for a simulation scenario"""
    name: str
    tvl_range: Tuple[float, float] = (50_000_000, 200_000_000)
    hack_probability_range: Tuple[float, float] = (0.05, 0.3)
    lgh_severity_range: Tuple[float, float] = (0.02, 0.5)
    market_stress_factor: float = 1.0  # Multiplier for risk aversion
    duration_days: int = 365
    num_simulations: int = 1000


class MarketSimulator:
    """Monte Carlo simulator for the DeFi insurance market"""
    
    def __init__(self, params: MarketParameters):
        self.params = params
        self.simulation_results = []
        
    def generate_hack_events(self, time_horizon: float, lambda_hack: float, 
                           num_simulations: int) -> List[List[Tuple[float, float]]]:
        """
        Generate hack events using Poisson process
        
        Args:
            time_horizon: Simulation time horizon in years
            lambda_hack: Hack intensity parameter
            num_simulations: Number of simulation paths
            
        Returns:
            List of hack event lists, each containing (time, severity) tuples
        """
        hack_events = []
        
        for _ in range(num_simulations):
            events = []
            t = 0
            
            while t < time_horizon:
                # Time to next hack (exponential distribution)
                inter_arrival = np.random.exponential(1 / lambda_hack)
                t += inter_arrival
                
                if t < time_horizon:
                    # Generate hack severity (LGH)
                    # Using beta distribution for realistic severity distribution
                    severity = np.random.beta(0.5, 2.0)  # Skewed towards smaller hacks
                    events.append((t, min(severity, 1.0)))
            
            hack_events.append(events)
        
        return hack_events
    
    def simulate_lgh_prices(self, base_prices: Dict[float, float],
                           volatility: float = 0.3,
                           time_steps: int = 365) -> List[Dict[float, float]]:
        """
        Simulate LGH token price evolution using geometric Brownian motion
        
        Args:
            base_prices: Initial LGH prices by strike
            volatility: Price volatility
            time_steps: Number of time steps
            
        Returns:
            List of price dictionaries over time
        """
        price_paths = []
        current_prices = base_prices.copy()
        
        dt = 1 / time_steps  # Daily time steps
        
        for _ in range(time_steps):
            new_prices = {}
            for strike, price in current_prices.items():
                # Geometric Brownian motion with mean reversion
                drift = -0.1 * (np.log(price) - np.log(base_prices[strike]))  # Mean reversion
                shock = volatility * np.sqrt(dt) * np.random.normal()
                
                new_price = price * np.exp((drift - 0.5 * volatility**2) * dt + shock)
                new_prices[strike] = max(0.001, min(0.1, new_price))  # Bounds
            
            current_prices = new_prices
            price_paths.append(current_prices.copy())
        
        return price_paths
    
    def simulate_tvl_evolution(self, initial_tvl: float, volatility: float = 0.4,
                              drift: float = 0.1, time_steps: int = 365) -> np.ndarray:
        """
        Simulate TVL evolution using geometric Brownian motion
        
        Args:
            initial_tvl: Starting TVL
            volatility: TVL volatility
            drift: TVL growth rate
            time_steps: Number of time steps
            
        Returns:
            Array of TVL values over time
        """
        dt = 1 / time_steps
        tvl_path = np.zeros(time_steps + 1)
        tvl_path[0] = initial_tvl
        
        for t in range(1, time_steps + 1):
            shock = np.random.normal()
            tvl_path[t] = tvl_path[t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shock
            )
            
            # Prevent TVL from going too low
            tvl_path[t] = max(tvl_path[t], initial_tvl * 0.1)
        
        return tvl_path
    
    def run_single_simulation(self, scenario: SimulationScenario, 
                             seed: int = None) -> Dict:
        """
        Run a single simulation path
        
        Args:
            scenario: Simulation scenario configuration
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize market
        market = InsuranceMarket(self.params)
        
        # Sample initial conditions
        initial_tvl = np.random.uniform(*scenario.tvl_range)
        hack_lambda = np.random.uniform(*scenario.hack_probability_range)
        
        # Generate time series
        time_horizon = scenario.duration_days / 365.0
        tvl_path = self.simulate_tvl_evolution(initial_tvl, time_steps=scenario.duration_days)
        
        # Generate hack events
        hack_events = self.generate_hack_events(time_horizon, hack_lambda, 1)[0]
        
        # Simulate LGH prices
        base_lgh_prices = {0.05: 0.008, 0.10: 0.015, 0.20: 0.025, 0.30: 0.035}
        lgh_price_path = self.simulate_lgh_prices(base_lgh_prices, time_steps=scenario.duration_days)
        
        # Track key metrics over time
        daily_results = []
        cumulative_losses = 0.0
        cumulative_payouts = 0.0
        
        # Find initial equilibrium
        market.tvl = initial_tvl
        market.update_lgh_prices(base_lgh_prices, {0.05: 1000, 0.10: 800, 0.20: 500, 0.30: 300})
        
        solver = EquilibriumSolver(market)
        try:
            eq_c_c, eq_c_lp, eq_c_spec = solver.find_equilibrium(max_iterations=30)
        except:
            # Fallback values if equilibrium not found
            eq_c_c = initial_tvl * 0.1
            eq_c_lp = initial_tvl * 0.5
            eq_c_spec = initial_tvl * 0.05
        
        market.c_c, market.c_lp, market.c_spec = eq_c_c, eq_c_lp, eq_c_spec
        
        # Daily simulation loop
        for day in range(scenario.duration_days):
            current_tvl = tvl_path[day]
            current_lgh_prices = lgh_price_path[day]
            
            # Update market state
            market.tvl = current_tvl
            market.update_lgh_prices(current_lgh_prices, {0.05: 1000, 0.10: 800, 0.20: 500, 0.30: 300})
            
            # Check for hack events on this day
            day_fraction = day / 365.0
            next_day_fraction = (day + 1) / 365.0
            
            daily_hack_loss = 0.0
            for hack_time, hack_severity in hack_events:
                if day_fraction <= hack_time < next_day_fraction:
                    # Hack occurred today
                    hack_loss = hack_severity * current_tvl
                    cumulative_losses += hack_loss
                    
                    # Calculate insurance payout
                    coverage = market.coverage_function(market.c_c, current_tvl)
                    insurance_payout = min(coverage, hack_loss)
                    cumulative_payouts += insurance_payout
                    
                    daily_hack_loss = hack_loss
                    
                    # Reduce LP capital by payout
                    market.c_lp = max(0, market.c_lp - insurance_payout)
            
            # Record daily state
            state = market.get_market_state()
            
            daily_results.append({
                'day': day,
                'tvl': current_tvl,
                'collateral': market.c_c,
                'lp_capital': market.c_lp,
                'speculator_capital': market.c_spec,
                'coverage': state['coverage'],
                'utilization': state['utilization'],
                'revenue_share': state['revenue_share'],
                'risk_price': state['risk_price'],
                'hack_loss': daily_hack_loss,
                'cumulative_losses': cumulative_losses,
                'cumulative_payouts': cumulative_payouts
            })
        
        # Calculate summary statistics
        total_hack_events = len(hack_events)
        total_hack_losses = cumulative_losses
        total_insurance_payouts = cumulative_payouts
        
        final_state = market.get_market_state()
        
        # Calculate profits
        p_hack = len(hack_events) / time_horizon if time_horizon > 0 else 0
        avg_lgh = np.mean([severity for _, severity in hack_events]) if hack_events else 0.1
        
        final_gamma = final_state['revenue_share']
        protocol_profit = market.protocol_profit(market.c_c, market.c_lp, market.c_spec, 
                                                p_hack, avg_lgh, final_gamma)
        lp_profit = market.lp_profit(market.c_c, market.c_lp, market.c_spec,
                                   p_hack, avg_lgh, final_gamma)
        
        return {
            'scenario': scenario.name,
            'initial_tvl': initial_tvl,
            'final_tvl': tvl_path[-1],
            'total_hack_events': total_hack_events,
            'total_hack_losses': total_hack_losses,
            'total_insurance_payouts': total_insurance_payouts,
            'final_lp_capital': market.c_lp,
            'final_utilization': final_state['utilization'],
            'final_revenue_share': final_state['revenue_share'],
            'protocol_profit': protocol_profit,
            'lp_profit': lp_profit,
            'hack_frequency': total_hack_events / time_horizon,
            'loss_ratio': total_insurance_payouts / max(total_hack_losses, 1),
            'daily_results': daily_results,
            'hack_events': hack_events
        }
    
    def run_monte_carlo(self, scenario: SimulationScenario) -> pd.DataFrame:
        """
        Run Monte Carlo simulation
        
        Args:
            scenario: Simulation scenario configuration
            
        Returns:
            DataFrame with simulation results
        """
        print(f"Running Monte Carlo simulation: {scenario.name}")
        print(f"Number of simulations: {scenario.num_simulations}")
        
        results = []
        
        for i in range(scenario.num_simulations):
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{scenario.num_simulations} simulations")
            
            try:
                result = self.run_single_simulation(scenario, seed=i)
                results.append(result)
            except Exception as e:
                print(f"Error in simulation {i}: {e}")
                continue
        
        self.simulation_results.extend(results)
        
        # Convert to DataFrame (excluding daily_results and hack_events for summary)
        summary_results = []
        for result in results:
            summary = {k: v for k, v in result.items() 
                      if k not in ['daily_results', 'hack_events']}
            summary_results.append(summary)
        
        return pd.DataFrame(summary_results)
    
    def stress_testing(self, base_scenario: SimulationScenario) -> Dict[str, pd.DataFrame]:
        """
        Perform stress testing with various scenarios
        
        Args:
            base_scenario: Base scenario configuration
            
        Returns:
            Dictionary of stress test results
        """
        stress_scenarios = {
            'base': base_scenario,
            'high_hack_frequency': SimulationScenario(
                name='High Hack Frequency',
                tvl_range=base_scenario.tvl_range,
                hack_probability_range=(0.2, 0.5),  # Higher hack rates
                lgh_severity_range=base_scenario.lgh_severity_range,
                duration_days=base_scenario.duration_days,
                num_simulations=base_scenario.num_simulations // 2
            ),
            'severe_hacks': SimulationScenario(
                name='Severe Hacks',
                tvl_range=base_scenario.tvl_range,
                hack_probability_range=base_scenario.hack_probability_range,
                lgh_severity_range=(0.1, 0.8),  # More severe hacks
                duration_days=base_scenario.duration_days,
                num_simulations=base_scenario.num_simulations // 2
            ),
            'market_crash': SimulationScenario(
                name='Market Crash',
                tvl_range=(20_000_000, 80_000_000),  # Lower TVL range
                hack_probability_range=(0.1, 0.4),  # Higher hack probability during stress
                lgh_severity_range=(0.05, 0.6),
                duration_days=base_scenario.duration_days,
                num_simulations=base_scenario.num_simulations // 2
            )
        }
        
        stress_results = {}
        
        for name, scenario in stress_scenarios.items():
            print(f"\nRunning stress test: {name}")
            results_df = self.run_monte_carlo(scenario)
            stress_results[name] = results_df
        
        return stress_results
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze simulation results and calculate key metrics
        
        Args:
            results_df: DataFrame with simulation results
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['summary_stats'] = {
            'num_simulations': len(results_df),
            'avg_hack_frequency': results_df['hack_frequency'].mean(),
            'avg_total_losses': results_df['total_hack_losses'].mean(),
            'avg_insurance_payouts': results_df['total_insurance_payouts'].mean(),
            'avg_loss_ratio': results_df['loss_ratio'].mean(),
        }
        
        # Profitability analysis
        analysis['profitability'] = {
            'protocol_profit_positive_pct': (results_df['protocol_profit'] > 0).mean() * 100,
            'lp_profit_positive_pct': (results_df['lp_profit'] > 0).mean() * 100,
            'avg_protocol_profit': results_df['protocol_profit'].mean(),
            'avg_lp_profit': results_df['lp_profit'].mean(),
        }
        
        # Risk metrics
        analysis['risk_metrics'] = {
            'var_95_hack_losses': results_df['total_hack_losses'].quantile(0.95),
            'var_99_hack_losses': results_df['total_hack_losses'].quantile(0.99),
            'max_hack_losses': results_df['total_hack_losses'].max(),
            'insolvency_probability': (results_df['final_lp_capital'] <= 0).mean() * 100,
        }
        
        # Utilization analysis
        analysis['utilization'] = {
            'avg_final_utilization': results_df['final_utilization'].mean(),
            'high_utilization_pct': (results_df['final_utilization'] > 1.5).mean() * 100,
            'extreme_utilization_pct': (results_df['final_utilization'] > 3.0).mean() * 100,
        }
        
        return analysis
    
    def plot_simulation_results(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot comprehensive simulation results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16)
        
        # 1. Hack frequency distribution
        axes[0, 0].hist(results_df['hack_frequency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Hack Frequency Distribution')
        axes[0, 0].set_xlabel('Hacks per Year')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Total losses distribution
        axes[0, 1].hist(results_df['total_hack_losses'] / 1e6, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Total Hack Losses Distribution')
        axes[0, 1].set_xlabel('Total Losses ($M)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Loss ratio distribution
        axes[0, 2].hist(results_df['loss_ratio'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Loss Ratio Distribution')
        axes[0, 2].set_xlabel('Insurance Payouts / Total Losses')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Protocol profit distribution
        axes[1, 0].hist(results_df['protocol_profit'] / 1e6, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', label='Break-even')
        axes[1, 0].set_title('Protocol Profit Distribution')
        axes[1, 0].set_xlabel('Protocol Profit ($M)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 5. LP profit distribution
        axes[1, 1].hist(results_df['lp_profit'] / 1e6, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', label='Break-even')
        axes[1, 1].set_title('LP Profit Distribution')
        axes[1, 1].set_xlabel('LP Profit ($M)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # 6. Final utilization distribution
        axes[1, 2].hist(results_df['final_utilization'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(x=1.0, color='red', linestyle='--', label='Full Utilization')
        axes[1, 2].set_title('Final Utilization Distribution')
        axes[1, 2].set_xlabel('Utilization Ratio')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        # 7. Hack losses vs Insurance payouts
        axes[2, 0].scatter(results_df['total_hack_losses'] / 1e6, 
                          results_df['total_insurance_payouts'] / 1e6, alpha=0.6)
        max_val = max(results_df['total_hack_losses'].max(), results_df['total_insurance_payouts'].max()) / 1e6
        axes[2, 0].plot([0, max_val], [0, max_val], 'r--', label='Perfect Coverage')
        axes[2, 0].set_title('Insurance Payouts vs Hack Losses')
        axes[2, 0].set_xlabel('Total Hack Losses ($M)')
        axes[2, 0].set_ylabel('Insurance Payouts ($M)')
        axes[2, 0].legend()
        
        # 8. Protocol vs LP profits
        axes[2, 1].scatter(results_df['protocol_profit'] / 1e6, 
                          results_df['lp_profit'] / 1e6, alpha=0.6)
        axes[2, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[2, 1].set_title('Protocol vs LP Profits')
        axes[2, 1].set_xlabel('Protocol Profit ($M)')
        axes[2, 1].set_ylabel('LP Profit ($M)')
        
        # 9. Final LP capital vs Initial TVL
        axes[2, 2].scatter(results_df['initial_tvl'] / 1e6, 
                          results_df['final_lp_capital'] / 1e6, alpha=0.6)
        axes[2, 2].set_title('Final LP Capital vs Initial TVL')
        axes[2, 2].set_xlabel('Initial TVL ($M)')
        axes[2, 2].set_ylabel('Final LP Capital ($M)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stress_test_comparison(self, stress_results: Dict[str, pd.DataFrame], 
                                  save_path: str = None):
        """Plot comparison of stress test results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Stress Test Comparison', fontsize=16)
        
        scenarios = list(stress_results.keys())
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # 1. Hack frequency comparison
        for i, (scenario, df) in enumerate(stress_results.items()):
            axes[0, 0].hist(df['hack_frequency'], bins=20, alpha=0.6, 
                           label=scenario, color=colors[i % len(colors)])
        axes[0, 0].set_title('Hack Frequency by Scenario')
        axes[0, 0].set_xlabel('Hacks per Year')
        axes[0, 0].legend()
        
        # 2. Total losses comparison
        for i, (scenario, df) in enumerate(stress_results.items()):
            axes[0, 1].hist(df['total_hack_losses'] / 1e6, bins=20, alpha=0.6,
                           label=scenario, color=colors[i % len(colors)])
        axes[0, 1].set_title('Total Losses by Scenario')
        axes[0, 1].set_xlabel('Total Losses ($M)')
        axes[0, 1].legend()
        
        # 3. Protocol profitability comparison
        profitability_data = []
        for scenario, df in stress_results.items():
            profit_positive_pct = (df['protocol_profit'] > 0).mean() * 100
            profitability_data.append(profit_positive_pct)
        
        axes[0, 2].bar(scenarios, profitability_data, color=colors[:len(scenarios)])
        axes[0, 2].set_title('Protocol Profitability by Scenario')
        axes[0, 2].set_ylabel('% Profitable Simulations')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. LP profitability comparison
        lp_profitability_data = []
        for scenario, df in stress_results.items():
            lp_profit_positive_pct = (df['lp_profit'] > 0).mean() * 100
            lp_profitability_data.append(lp_profit_positive_pct)
        
        axes[1, 0].bar(scenarios, lp_profitability_data, color=colors[:len(scenarios)])
        axes[1, 0].set_title('LP Profitability by Scenario')
        axes[1, 0].set_ylabel('% Profitable Simulations')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Insolvency risk comparison
        insolvency_data = []
        for scenario, df in stress_results.items():
            insolvency_pct = (df['final_lp_capital'] <= 0).mean() * 100
            insolvency_data.append(insolvency_pct)
        
        axes[1, 1].bar(scenarios, insolvency_data, color=colors[:len(scenarios)])
        axes[1, 1].set_title('Insolvency Risk by Scenario')
        axes[1, 1].set_ylabel('% Insolvent Simulations')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Average utilization comparison
        avg_utilization_data = []
        for scenario, df in stress_results.items():
            avg_util = df['final_utilization'].mean()
            avg_utilization_data.append(avg_util)
        
        axes[1, 2].bar(scenarios, avg_utilization_data, color=colors[:len(scenarios)])
        axes[1, 2].axhline(y=1.0, color='red', linestyle='--', label='Full Utilization')
        axes[1, 2].set_title('Average Final Utilization')
        axes[1, 2].set_ylabel('Utilization Ratio')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_simulation_report(self, results_df: pd.DataFrame, 
                                 analysis: Dict) -> str:
        """Generate comprehensive simulation report"""
        
        report = f"""
DeFi Insurance Market Simulation Report
======================================

Simulation Overview:
-------------------
Number of Simulations: {analysis['summary_stats']['num_simulations']:,}
Average Hack Frequency: {analysis['summary_stats']['avg_hack_frequency']:.3f} per year
Average Total Losses: ${analysis['summary_stats']['avg_total_losses']:,.0f}
Average Insurance Payouts: ${analysis['summary_stats']['avg_insurance_payouts']:,.0f}
Average Loss Ratio: {analysis['summary_stats']['avg_loss_ratio']:.3f}

Profitability Analysis:
----------------------
Protocol Profitability:
  - % Profitable Simulations: {analysis['profitability']['protocol_profit_positive_pct']:.1f}%
  - Average Profit: ${analysis['profitability']['avg_protocol_profit']:,.0f}

LP Profitability:
  - % Profitable Simulations: {analysis['profitability']['lp_profit_positive_pct']:.1f}%
  - Average Profit: ${analysis['profitability']['avg_lp_profit']:,.0f}

Risk Assessment:
---------------
Value at Risk (95%): ${analysis['risk_metrics']['var_95_hack_losses']:,.0f}
Value at Risk (99%): ${analysis['risk_metrics']['var_99_hack_losses']:,.0f}
Maximum Observed Loss: ${analysis['risk_metrics']['max_hack_losses']:,.0f}
Insolvency Probability: {analysis['risk_metrics']['insolvency_probability']:.2f}%

Market Utilization:
------------------
Average Final Utilization: {analysis['utilization']['avg_final_utilization']:.2f}
High Utilization (>1.5): {analysis['utilization']['high_utilization_pct']:.1f}% of simulations
Extreme Utilization (>3.0): {analysis['utilization']['extreme_utilization_pct']:.1f}% of simulations

Key Insights:
------------
"""
        
        # Add insights based on results
        if analysis['profitability']['protocol_profit_positive_pct'] > 80:
            report += "✓ Protocol model shows strong profitability across scenarios\n"
        elif analysis['profitability']['protocol_profit_positive_pct'] > 60:
            report += "⚠ Protocol profitability is moderate, consider parameter adjustments\n"
        else:
            report += "⚠ Protocol profitability is poor, significant model changes needed\n"
        
        if analysis['profitability']['lp_profit_positive_pct'] > 70:
            report += "✓ LP model provides attractive risk-adjusted returns\n"
        else:
            report += "⚠ LP returns may be insufficient to attract capital\n"
        
        if analysis['risk_metrics']['insolvency_probability'] < 5:
            report += "✓ Low insolvency risk indicates robust capital structure\n"
        elif analysis['risk_metrics']['insolvency_probability'] < 15:
            report += "⚠ Moderate insolvency risk requires monitoring\n"
        else:
            report += "⚠ High insolvency risk requires immediate attention\n"
        
        if analysis['utilization']['avg_final_utilization'] < 1.2:
            report += "✓ Utilization levels are sustainable\n"
        else:
            report += "⚠ High utilization levels may strain the system\n"
        
        return report


def run_comprehensive_simulation():
    """Run comprehensive simulation analysis"""
    print("Starting Comprehensive DeFi Insurance Simulation")
    print("=" * 55)
    
    # Use base parameters (could be optimized parameters from optimization module)
    params = MarketParameters(
        mu=1000.0,
        theta=0.6,
        xi=0.15,
        alpha=0.7,
        beta=1.8,
        delta=1.5,
        u_target=0.2,
        r_market=0.05,
        r_pool=0.08,
        rho=0.025,
        lambda_hack=0.12
    )
    
    # Initialize simulator
    simulator = MarketSimulator(params)
    
    # Define base scenario
    base_scenario = SimulationScenario(
        name='Base Case',
        tvl_range=(50_000_000, 200_000_000),
        hack_probability_range=(0.05, 0.25),
        lgh_severity_range=(0.02, 0.4),
        duration_days=365,
        num_simulations=500
    )
    
    # Run base simulation
    print("Running base scenario simulation...")
    base_results = simulator.run_monte_carlo(base_scenario)
    
    # Analyze base results
    base_analysis = simulator.analyze_results(base_results)
    
    # Run stress testing
    print("\nRunning stress tests...")
    stress_results = simulator.stress_testing(base_scenario)
    
    # Generate reports
    print("\n" + simulator.generate_simulation_report(base_results, base_analysis))
    
    # Print stress test summary
    print("\nStress Test Summary:")
    print("=" * 20)
    for scenario_name, df in stress_results.items():
        analysis = simulator.analyze_results(df)
        print(f"\n{scenario_name}:")
        print(f"  Protocol Profitability: {analysis['profitability']['protocol_profit_positive_pct']:.1f}%")
        print(f"  LP Profitability: {analysis['profitability']['lp_profit_positive_pct']:.1f}%")
        print(f"  Insolvency Risk: {analysis['risk_metrics']['insolvency_probability']:.2f}%")
    
    # Plot results
    print("\nGenerating plots...")
    simulator.plot_simulation_results(base_results)
    simulator.plot_stress_test_comparison(stress_results)
    
    return simulator, base_results, stress_results


if __name__ == "__main__":
    simulator, base_results, stress_results = run_comprehensive_simulation()
