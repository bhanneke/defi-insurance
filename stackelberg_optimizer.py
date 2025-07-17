"""
Automated Stackelberg Game Optimization Framework
Iterates the game across different parameters and market specifications automatically
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from stackelberg_operator_game import StackelbergInsuranceGame, MarketState, OperatorParameters

# ===========================================================================
# AUTOMATED OPTIMIZATION FRAMEWORK
# ===========================================================================

@dataclass
class OptimizationSpec:
    """Specification for automated optimization"""
    # NOTE: r_pool is now exogenous and should not be included in the optimization
    delta_r_values: List[float] = (-0.01, 0.0, 0.01, 0.02)  # r_market - r_pool deltas to simulate
        # Parameter ranges to test
    phi_range: Tuple[float, float] = (0.001, 0.15)          # Operator fee range
    mu_range: Tuple[float, float] = (500.0, 3000.0)        # Coverage amplification
    theta_range: Tuple[float, float] = (0.3, 0.8)          # Coverage concavity
    xi_range: Tuple[float, float] = (0.05, 0.4)            # Security scaling
    alpha_range: Tuple[float, float] = (0.2, 0.9)          # Utilization weight
    beta_range: Tuple[float, float] = (0.8, 3.0)           # Utilization convexity
    delta_range: Tuple[float, float] = (0.8, 3.0)          # Risk price convexity
    u_target_range: Tuple[float, float] = (0.1, 0.8)       # Target utilization
        
    # Market scenario ranges
    tvl_range: Tuple[float, float] = (50_000_000, 500_000_000)     # TVL range
    p_hack_range: Tuple[float, float] = (0.05, 0.3)                # Hack probability
    expected_lgh_range: Tuple[float, float] = (0.03, 0.3)          # Expected loss
    r_market_range: Tuple[float, float] = (0.03, 0.08)             # Market rate
    
    # Optimization settings
    num_parameter_samples: int = 1000                       # Parameter combinations to test
    num_market_scenarios: int = 50                          # Market scenarios per parameter set
    parallel_workers: int = 4                               # Parallel processing workers
    
    # Evaluation criteria weights
    operator_profit_weight: float = 0.2
    market_viability_weight: float = 0.3
    system_stability_weight: float = 0.25
    stakeholder_welfare_weight: float = 0.25

@dataclass
class OptimizationResult:
    """Result from automated optimization"""
    parameter_set: OperatorParameters
    market_scenario: MarketState
    game_outcome: Dict
    evaluation_score: float
    success: bool
    metrics: Dict

class AutomatedGameOptimizer:
    """
    Automated optimization framework that runs the Stackelberg game
    across different parameters and market specifications
    """
    
    def __init__(self, spec: OptimizationSpec):
        self.spec = spec
        self.results: List[OptimizationResult] = []
        self.best_parameters: Optional[OperatorParameters] = None
        self.optimization_history = pd.DataFrame()
        
    def generate_parameter_samples(self, method: str = "random") -> List[OperatorParameters]:
        """Generate parameter combinations to test"""
        
        if method == "random":
            # Random sampling within ranges
            samples = []
            for _ in range(self.spec.num_parameter_samples):
                params = OperatorParameters(
                    phi=np.random.uniform(*self.spec.phi_range),
                    mu=np.random.uniform(*self.spec.mu_range),
                    theta=np.random.uniform(*self.spec.theta_range),
                    xi=np.random.uniform(*self.spec.xi_range),
                    alpha=np.random.uniform(*self.spec.alpha_range),
                    beta=np.random.uniform(*self.spec.beta_range),
                    delta=np.random.uniform(*self.spec.delta_range),
                    u_target=np.random.uniform(*self.spec.u_target_range),
                                    )
                samples.append(params)
                
        elif method == "grid":
            # Grid sampling (smaller grid due to computational cost)
            n_per_dim = int(self.spec.num_parameter_samples ** (1/9))  # 9 parameters
            
            phi_vals = np.linspace(*self.spec.phi_range, n_per_dim)
            mu_vals = np.linspace(*self.spec.mu_range, n_per_dim)
            theta_vals = np.linspace(*self.spec.theta_range, n_per_dim)
            alpha_vals = np.linspace(*self.spec.alpha_range, n_per_dim)
            
            # Use subset for computational feasibility
            samples = []
            for phi, mu, theta, alpha in product(phi_vals[:3], mu_vals[:3], theta_vals[:3], alpha_vals[:3]):
                params = OperatorParameters(
                    phi=phi, mu=mu, theta=theta, xi=0.2, alpha=alpha,
                    beta=1.5, delta=1.3, u_target=0.4, 
                                    )
                samples.append(params)
                if len(samples) >= self.spec.num_parameter_samples:
                    break
                    
        elif method == "sobol":
            # Sobol sequence for better space coverage
            from scipy.stats import qmc
            
            sampler = qmc.Sobol(d=9, scramble=True)
            sobol_samples = sampler.random(n=self.spec.num_parameter_samples)
            
            samples = []
            for sample in sobol_samples:
                params = OperatorParameters(
                    phi=self.spec.phi_range[0] + sample[0] * (self.spec.phi_range[1] - self.spec.phi_range[0]),
                    mu=self.spec.mu_range[0] + sample[1] * (self.spec.mu_range[1] - self.spec.mu_range[0]),
                    theta=self.spec.theta_range[0] + sample[2] * (self.spec.theta_range[1] - self.spec.theta_range[0]),
                    xi=self.spec.xi_range[0] + sample[3] * (self.spec.xi_range[1] - self.spec.xi_range[0]),
                    alpha=self.spec.alpha_range[0] + sample[4] * (self.spec.alpha_range[1] - self.spec.alpha_range[0]),
                    beta=self.spec.beta_range[0] + sample[5] * (self.spec.beta_range[1] - self.spec.beta_range[0]),
                    delta=self.spec.delta_range[0] + sample[6] * (self.spec.delta_range[1] - self.spec.delta_range[0]),
                    u_target=self.spec.u_target_range[0] + sample[7] * (self.spec.u_target_range[1] - self.spec.u_target_range[0]),
                                    )
                samples.append(params)
        
        return samples
    
    def generate_market_scenarios(self) -> List[MarketState]:
        """Generate diverse market scenarios to test"""
        
        scenarios = []
        for _ in range(self.spec.num_market_scenarios):
            scenario = MarketState(
                tvl=np.random.uniform(*self.spec.tvl_range),
                c_c=0,  # Will be optimized
                c_lp=0,  # Will be optimized
                c_spec=np.random.uniform(1_000_000, 10_000_000),  # Random speculator capital
                c_premium=0,  # Will be calculated based on TVL
                p_hack=np.random.uniform(*self.spec.p_hack_range),
                expected_lgh=np.random.uniform(*self.spec.expected_lgh_range),
                p_risk=np.random.uniform(0.005, 0.04),  # LGH token prices
                r_market=np.random.uniform(*self.spec.r_market_range)
            )
            # Set premium as percentage of TVL
            scenario.c_premium = scenario.tvl * np.random.uniform(0.005, 0.02)  # 0.5-2% of TVL
            scenarios.append(scenario)
        
        return scenarios
    
    def evaluate_parameter_set(self, params: OperatorParameters, scenarios: List[MarketState]) -> Dict:
        """Evaluate a parameter set across multiple market scenarios"""
        
        scenario_results = []
        
        for scenario in scenarios:
            try:
                # Run the Stackelberg game
                game = StackelbergInsuranceGame(scenario)
                
                # Quick feasibility check
                if (params.r_pool <= scenario.r_market or 
                    params.theta <= 0.1 or params.theta >= 1.0 or
                    params.phi <= 0 or params.phi >= 0.5):
                    scenario_results.append({'success': False, 'reason': 'infeasible_parameters'})
                    continue
                
                # Evaluate the parameters
                result = game.evaluate_operator_parameters(params)
                scenario_results.append(result)
                
            except Exception as e:
                scenario_results.append({'success': False, 'error': str(e)})
        
        # Aggregate results across scenarios
        successful_results = [r for r in scenario_results if r.get('success', False)]
        
        if len(successful_results) == 0:
            return {
                'success': False,
                'success_rate': 0.0,
                'avg_operator_profit': 0,
                'avg_market_viability': 0,
                'avg_system_stability': 0,
                'avg_stakeholder_welfare': 0,
                'overall_score': 0
            }
        
        success_rate = len(successful_results) / len(scenarios)
        
        # Calculate aggregate metrics
        operator_profits = [r['utilities']['operator'] for r in successful_results]
        market_viabilities = [r['participation']['both_participate'] for r in successful_results]
        utilizations = [r['market_metrics']['utilization'] for r in successful_results]
        protocol_utilities = [r['utilities']['protocol'] for r in successful_results]
        lp_utilities = [r['utilities']['lp'] for r in successful_results]
        
        avg_operator_profit = np.mean(operator_profits)
        avg_market_viability = np.mean(market_viabilities)
        avg_system_stability = 1.0 / (1.0 + np.std(utilizations))  # Lower volatility = higher stability
        avg_stakeholder_welfare = np.mean(protocol_utilities) + np.mean(lp_utilities)
        
        # Calculate overall score using weights
        overall_score = (
            self.spec.operator_profit_weight * min(1.0, avg_operator_profit / 1_000_000) +  # Normalize to $1M
            self.spec.market_viability_weight * avg_market_viability +
            self.spec.system_stability_weight * avg_system_stability +
            self.spec.stakeholder_welfare_weight * min(1.0, avg_stakeholder_welfare / 1_000_000)  # Normalize
        )
        
        return {
            'success': True,
            'success_rate': success_rate,
            'avg_operator_profit': avg_operator_profit,
            'avg_market_viability': avg_market_viability,
            'avg_system_stability': avg_system_stability,
            'avg_stakeholder_welfare': avg_stakeholder_welfare,
            'overall_score': overall_score,
            'detailed_results': successful_results
        }
    
    def run_automated_optimization(self, sampling_method: str = "sobol") -> pd.DataFrame:
        """
        Run the complete automated optimization
        """
        
        print("🚀 STARTING AUTOMATED STACKELBERG GAME OPTIMIZATION")
        print("=" * 60)
        print(f"Parameter samples: {self.spec.num_parameter_samples}")
        print(f"Market scenarios per sample: {self.spec.num_market_scenarios}")
        print(f"Total game evaluations: {self.spec.num_parameter_samples * self.spec.num_market_scenarios}")
        print(f"Sampling method: {sampling_method}")
        print(f"Parallel workers: {self.spec.parallel_workers}")
        print()
        
        # Generate parameter samples and market scenarios
        print("🎯 Generating parameter samples...")
        parameter_samples = self.generate_parameter_samples(sampling_method)
        
        print("🌍 Generating market scenarios...")
        market_scenarios = self.generate_market_scenarios()
        
        # Run optimization
        print("⚡ Running automated optimization...")
        start_time = time.time()
        
        results = []
        completed = 0
        
        if self.spec.parallel_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.spec.parallel_workers) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(self.evaluate_parameter_set, params, market_scenarios): params 
                    for params in parameter_samples
                }
                
                # Collect results
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        evaluation = future.result()
                        results.append({
                            'parameters': asdict(params),
                            'evaluation': evaluation
                        })
                        completed += 1
                        
                        if completed % 50 == 0:
                            print(f"   Progress: {completed}/{len(parameter_samples)} ({completed/len(parameter_samples)*100:.1f}%)")
                            
                    except Exception as e:
                        print(f"   ❌ Parameter evaluation failed: {e}")
        else:
            # Sequential execution
            for i, params in enumerate(parameter_samples):
                evaluation = self.evaluate_parameter_set(params, market_scenarios)
                results.append({
                    'parameters': asdict(params),
                    'evaluation': evaluation
                })
                
                if (i + 1) % 50 == 0:
                    print(f"   Progress: {i+1}/{len(parameter_samples)} ({(i+1)/len(parameter_samples)*100:.1f}%)")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Optimization completed in {elapsed_time:.1f} seconds")
        
        # Process results into DataFrame
        processed_results = []
        for result in results:
            if result['evaluation']['success']:
                row = {}
                # Add parameters
                for key, value in result['parameters'].items():
                    row[f'param_{key}'] = value
                
                # Add evaluation metrics
                for key, value in result['evaluation'].items():
                    if key != 'detailed_results':
                        row[f'eval_{key}'] = value
                
                processed_results.append(row)
        
        self.optimization_history = pd.DataFrame(processed_results)
        
        print(f"📊 Successful evaluations: {len(processed_results)}/{len(parameter_samples)} ({len(processed_results)/len(parameter_samples)*100:.1f}%)")
        
        return self.optimization_history
    
    def analyze_results(self) -> Dict:
        """Analyze optimization results and identify best parameters"""
        
        if self.optimization_history.empty:
            print("❌ No results to analyze. Run optimization first.")
            return {}
        
        print("\n📊 ANALYZING OPTIMIZATION RESULTS")
        print("=" * 40)
        
        df = self.optimization_history
        
        # Find best parameter set
        best_idx = df['eval_overall_score'].idxmax()
        best_row = df.loc[best_idx]
        
        self.best_parameters = OperatorParameters(
            phi=best_row['param_phi'],
            mu=best_row['param_mu'],
            theta=best_row['param_theta'],
            xi=best_row['param_xi'],
            alpha=best_row['param_alpha'],
            beta=best_row['param_beta'],
            delta=best_row['param_delta'],
            u_target=best_row['param_u_target'],
            r_pool=best_row['param_r_pool']
        )
        
        print("🏆 BEST PARAMETER SET:")
        print(f"   φ (operator fee): {self.best_parameters.phi:.3f} ({self.best_parameters.phi*100:.1f}%)")
        print(f"   μ (coverage factor): {self.best_parameters.mu:.0f}")
        print(f"   θ (concavity): {self.best_parameters.theta:.3f}")
        print(f"   α (utilization weight): {self.best_parameters.alpha:.3f}")
        print(f"   u_target: {self.best_parameters.u_target:.3f}")
                
        print(f"\n📈 BEST PERFORMANCE METRICS:")
        print(f"   Overall score: {best_row['eval_overall_score']:.3f}")
        print(f"   Success rate: {best_row['eval_success_rate']:.1%}")
        print(f"   Avg operator profit: ${best_row['eval_avg_operator_profit']:,.0f}")
        print(f"   Market viability: {best_row['eval_avg_market_viability']:.1%}")
        print(f"   System stability: {best_row['eval_avg_system_stability']:.3f}")
        
        # Summary statistics
        summary_stats = {
            'total_evaluations': len(df),
            'best_overall_score': best_row['eval_overall_score'],
            'best_parameters': asdict(self.best_parameters),
            'performance_distribution': {
                'score_mean': df['eval_overall_score'].mean(),
                'score_std': df['eval_overall_score'].std(),
                'score_95th_percentile': df['eval_overall_score'].quantile(0.95)
            },
            'parameter_insights': self._analyze_parameter_relationships()
        }
        
        return summary_stats
    
    def _analyze_parameter_relationships(self) -> Dict:
        """Analyze relationships between parameters and performance"""
        
        df = self.optimization_history
        
        # Calculate correlations
        param_cols = [col for col in df.columns if col.startswith('param_')]
        score_col = 'eval_overall_score'
        
        correlations = {}
        for col in param_cols:
            param_name = col.replace('param_', '')
            corr = df[col].corr(df[score_col])
            correlations[param_name] = corr
        
        # Find most important parameters
        important_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'parameter_correlations': correlations,
            'most_important_parameters': important_params[:5],
            'parameter_ranges_top_10pct': self._get_top_parameter_ranges(df, 0.1)
        }
    
    def _get_top_parameter_ranges(self, df: pd.DataFrame, percentile: float) -> Dict:
        """Get parameter ranges for top performing parameter sets"""
        
        threshold = df['eval_overall_score'].quantile(1 - percentile)
        top_df = df[df['eval_overall_score'] >= threshold]
        
        ranges = {}
        param_cols = [col for col in df.columns if col.startswith('param_')]
        
        for col in param_cols:
            param_name = col.replace('param_', '')
            ranges[param_name] = {
                'min': top_df[col].min(),
                'max': top_df[col].max(),
                'mean': top_df[col].mean(),
                'std': top_df[col].std()
            }
        
        return ranges
    
    def plot_optimization_results(self, save_path: Optional[str] = None):
        """Create comprehensive visualization of optimization results"""
        
        if self.optimization_history.empty:
            print("❌ No results to plot")
            return
        
        df = self.optimization_history
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Automated Stackelberg Game Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Overall score distribution
        axes[0, 0].hist(df['eval_overall_score'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['eval_overall_score'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].set_xlabel('Overall Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Success rate vs Overall score
        axes[0, 1].scatter(df['eval_success_rate'], df['eval_overall_score'], alpha=0.6)
        axes[0, 1].set_title('Success Rate vs Overall Score')
        axes[0, 1].set_xlabel('Success Rate')
        axes[0, 1].set_ylabel('Overall Score')
        
        # 3. Operator profit vs Market viability
        axes[0, 2].scatter(df['eval_avg_operator_profit']/1000, df['eval_avg_market_viability'], 
                          c=df['eval_overall_score'], cmap='viridis', alpha=0.7)
        axes[0, 2].set_title('Operator Profit vs Market Viability')
        axes[0, 2].set_xlabel('Avg Operator Profit ($K)')
        axes[0, 2].set_ylabel('Market Viability Rate')
        
        # 4-6. Key parameter relationships
        key_params = ['param_phi', 'param_alpha', 'param_mu']
        for i, param in enumerate(key_params):
            ax = axes[1, i]
            ax.scatter(df[param], df['eval_overall_score'], alpha=0.6)
            ax.set_title(f'{param.replace("param_", "").title()} vs Performance')
            ax.set_xlabel(param.replace('param_', '').title())
            ax.set_ylabel('Overall Score')
        
        # 7. Parameter correlation heatmap
        param_cols = [col for col in df.columns if col.startswith('param_')]
        corr_matrix = df[param_cols + ['eval_overall_score']].corr()
        
        im = axes[2, 0].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        axes[2, 0].set_title('Parameter Correlation Matrix')
        param_labels = [col.replace('param_', '') for col in param_cols] + ['score']
        axes[2, 0].set_xticks(range(len(param_labels)))
        axes[2, 0].set_yticks(range(len(param_labels)))
        axes[2, 0].set_xticklabels(param_labels, rotation=45)
        axes[2, 0].set_yticklabels(param_labels)
        plt.colorbar(im, ax=axes[2, 0])
        
        # 8. Top 10% parameter ranges
        top_10_pct = df['eval_overall_score'].quantile(0.9)
        top_df = df[df['eval_overall_score'] >= top_10_pct]
        
        performance_metrics = ['eval_avg_operator_profit', 'eval_avg_market_viability', 'eval_avg_system_stability']
        metric_names = ['Operator Profit', 'Market Viability', 'System Stability']
        
        for i, (metric, name) in enumerate(zip(performance_metrics, metric_names)):
            if i == 0:  # Operator profit
                axes[2, 1].hist(top_df[metric]/1000, bins=20, alpha=0.7, label=name)
                axes[2, 1].set_xlabel('Value ($K for profit)')
            elif i == 1:  # Market viability  
                axes[2, 1].hist(top_df[metric], bins=20, alpha=0.7, label=name)
            else:  # System stability
                axes[2, 2].hist(top_df[metric], bins=20, alpha=0.7, label=name)
                axes[2, 2].set_xlabel('Stability Score')
        
        axes[2, 1].set_title('Top 10% Performance Distribution')
        axes[2, 1].legend()
        axes[2, 2].set_title('Top 10% Stability Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_automated_optimization_example():
    """Example of running the automated optimization"""
    
    print("🤖 AUTOMATED STACKELBERG GAME OPTIMIZATION EXAMPLE")
    print("=" * 55)
    
    # Define optimization specification
    spec = OptimizationSpec(
        # Reasonable parameter ranges based on your paper
        phi_range=(0.01, 0.12),           # 1-12% operator fee
        mu_range=(800.0, 2500.0),         # Coverage amplification
        alpha_range=(0.3, 0.8),           # Utilization weight  
        u_target_range=(0.2, 0.7),        # Target utilization
        
        # Test settings (adjust for speed vs thoroughness)
        num_parameter_samples=200,         # Reduce for faster testing
        num_market_scenarios=20,           # Reduce for faster testing
        parallel_workers=2,                # Adjust based on your system
        
        # Evaluation weights
        operator_profit_weight=0.25,
        market_viability_weight=0.35,     # Higher weight on market functioning
        system_stability_weight=0.25,
        stakeholder_welfare_weight=0.15
    )
    
    # Run optimization
    optimizer = AutomatedGameOptimizer(spec)
    
    print("Starting automated optimization...")
    results_df = optimizer.run_automated_optimization(sampling_method="sobol")
    
    # Analyze results
    analysis = optimizer.analyze_results()
    
    # Visualize results
    optimizer.plot_optimization_results()
    
    print("\n🎯 OPTIMIZATION SUMMARY:")
    print(f"   Total successful evaluations: {analysis['total_evaluations']}")
    print(f"   Best overall score: {analysis['best_overall_score']:.3f}")
    
    return optimizer, results_df, analysis


if __name__ == "__main__":
    # Run the automated optimization
    optimizer, results, analysis = run_automated_optimization_example()
    
    print("\n✅ Automated optimization completed!")
    print("The system has automatically:")
    print("   🎯 Tested hundreds of parameter combinations")
    print("   🌍 Across dozens of market scenarios")  
    print("   🎮 Running the full Stackelberg game each time")
    print("   📊 Identified the best-performing parameters")
    print("   📈 Provided comprehensive analysis and visualization")
