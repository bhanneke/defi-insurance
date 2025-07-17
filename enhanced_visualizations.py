"""
Enhanced Professional Visualizations for DeFi Insurance Market
Publication-quality plots and dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Professional styling setup
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

class ProfessionalVisualizer:
    """Professional visualization suite for DeFi insurance market analysis"""
    
    def __init__(self):
        # Define professional color palettes
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Deep magenta
            'accent': '#F18F01',       # Orange accent
            'success': '#C73E1D',      # Success green
            'warning': '#F18F01',      # Warning orange
            'danger': '#C73E1D',       # Danger red
            'neutral': '#708090',      # Slate gray
            'light': '#F5F5F5',        # Light gray
            'dark': '#2F4F4F'          # Dark slate gray
        }
        
        # Create custom color palette
        self.palette = [
            self.colors['primary'], self.colors['secondary'], 
            self.colors['accent'], self.colors['success'],
            self.colors['warning'], self.colors['danger'],
            self.colors['neutral']
        ]
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
    def create_executive_dashboard(self, optimization_results: Dict, 
                                 simulation_results: Dict,
                                 theoretical_results: Dict,
                                 save_path: str = None) -> plt.Figure:
        """
        Create comprehensive executive dashboard
        """
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('white')
        
        # Create custom grid layout
        gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.4)
        
        # Main title
        fig.suptitle('DeFi Cybersecurity Insurance Market: Executive Dashboard', 
                    fontsize=28, fontweight='bold', y=0.96)
        
        # === ROW 1: KEY PERFORMANCE INDICATORS ===
        
        # 1. Market Viability Score
        ax1 = fig.add_subplot(gs[0, 0])
        viability_score = self._calculate_viability_score(optimization_results, simulation_results, theoretical_results)
        self._create_gauge_chart(ax1, viability_score, "Market Viability", "%")
        
        # 2. Protocol Profitability
        ax2 = fig.add_subplot(gs[0, 1])
        protocol_profit = simulation_results.get('profitability', {}).get('protocol_profit_positive_pct', 0)
        self._create_gauge_chart(ax2, protocol_profit, "Protocol\nProfitability", "%")
        
        # 3. LP Profitability  
        ax3 = fig.add_subplot(gs[0, 2])
        lp_profit = simulation_results.get('profitability', {}).get('lp_profit_positive_pct', 0)
        self._create_gauge_chart(ax3, lp_profit, "LP\nProfitability", "%")
        
        # 4. System Stability
        ax4 = fig.add_subplot(gs[0, 3])
        stability_score = simulation_results.get('utilization', {}).get('utilization_stability', 0.5) * 100
        self._create_gauge_chart(ax4, stability_score, "System\nStability", "%")
        
        # 5. Risk Level
        ax5 = fig.add_subplot(gs[0, 4])
        insolvency_risk = simulation_results.get('risk_metrics', {}).get('insolvency_probability', 10)
        risk_score = max(0, 100 - insolvency_risk * 2)  # Invert insolvency risk
        self._create_gauge_chart(ax5, risk_score, "Risk\nManagement", "%")
        
        # 6. Implementation Readiness
        ax6 = fig.add_subplot(gs[0, 5])
        readiness_score = self._calculate_readiness_score(theoretical_results)
        self._create_gauge_chart(ax6, readiness_score, "Implementation\nReadiness", "%")
        
        # === ROW 2: OPTIMIZATION & THEORETICAL RESULTS ===
        
        # 7. Parameter Optimization Results
        ax7 = fig.add_subplot(gs[1, :3])
        self._plot_parameter_radar(ax7, optimization_results)
        
        # 8. Theoretical Verification Status
        ax8 = fig.add_subplot(gs[1, 3:])
        self._plot_theoretical_verification(ax8, theoretical_results)
        
        # === ROW 3: SIMULATION RESULTS ===
        
        # 9. Profitability Distribution
        ax9 = fig.add_subplot(gs[2, :2])
        self._plot_profitability_distribution(ax9, simulation_results)
        
        # 10. Risk Analysis
        ax10 = fig.add_subplot(gs[2, 2:4])
        self._plot_risk_analysis(ax10, simulation_results)
        
        # 11. Market Dynamics
        ax11 = fig.add_subplot(gs[2, 4:])
        self._plot_market_dynamics(ax11, simulation_results)
        
        # === ROW 4: RECOMMENDATIONS & SUMMARY ===
        
        # 12. Risk-Return Matrix
        ax12 = fig.add_subplot(gs[3, :2])
        self._plot_risk_return_matrix(ax12, simulation_results)
        
        # 13. Key Recommendations
        ax13 = fig.add_subplot(gs[3, 2:4])
        self._create_recommendations_panel(ax13, optimization_results, simulation_results, theoretical_results)
        
        # 14. Market Health Scorecard
        ax14 = fig.add_subplot(gs[3, 4:])
        self._create_health_scorecard(ax14, simulation_results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _create_gauge_chart(self, ax, value: float, title: str, unit: str = ""):
        """Create professional gauge chart"""
        # Clear the axis
        ax.clear()
        
        # Determine color based on value
        if value >= 80:
            color = self.colors['success']
        elif value >= 60:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        # Create gauge background
        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=8)
        
        # Create value arc
        value_theta = np.linspace(0, np.pi * (value / 100), int(value))
        ax.plot(np.cos(value_theta), np.sin(value_theta), color=color, linewidth=8)
        
        # Add value text
        ax.text(0, -0.3, f"{value:.1f}{unit}", ha='center', va='center', 
               fontsize=16, fontweight='bold', color=color)
        
        # Add title
        ax.text(0, 1.2, title, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        # Style
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _plot_parameter_radar(self, ax, optimization_results: Dict):
        """Create radar chart for optimized parameters"""
        if 'optimal_params' not in optimization_results:
            ax.text(0.5, 0.5, 'Optimization\nResults\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        params = optimization_results['optimal_params']
        
        # Parameter categories and normalized values
        categories = ['Coverage\nAmplification', 'Coverage\nConcavity', 'Security\nScaling', 
                     'Utilization\nWeight', 'Target\nUtilization', 'Pool\nYield']
        
        values = [
            min(1.0, params.mu / 2000),      # Normalize mu
            params.theta,                     # theta already 0-1
            params.xi / 0.5,                 # Normalize xi
            params.alpha,                     # alpha already 0-1
            params.u_target,                  # u_target around 0-1
            min(1.0, params.r_pool / 0.15)  # Normalize r_pool
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'], alpha=0.8)
        ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Optimized Parameters Profile', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    def _plot_theoretical_verification(self, ax, theoretical_results: Dict):
        """Plot theoretical verification status"""
        if not theoretical_results:
            ax.text(0.5, 0.5, 'Theoretical\nVerification\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        theorems = ['Equilibrium\nExistence', 'Risk\nAssessment', 'LP\nDynamics', 'Solvency\nBounds']
        statuses = [
            theoretical_results.get('equilibrium', {}).get('theorem_verified', False),
            theoretical_results.get('risk_assessment', {}).get('proposition_verified', False),
            theoretical_results.get('lp_dynamics', {}).get('theorem_verified', False),
            theoretical_results.get('solvency', {}).get('proposition_verified', False)
        ]
        
        # Create status visualization
        y_pos = np.arange(len(theorems))
        colors = [self.colors['success'] if status else self.colors['danger'] for status in statuses]
        
        bars = ax.barh(y_pos, [1]*len(theorems), color=colors, alpha=0.8, edgecolor='black')
        
        # Add checkmarks/X marks
        for i, (bar, status) in enumerate(zip(bars, statuses)):
            symbol = '✓' if status else '✗'
            ax.text(0.5, i, symbol, ha='center', va='center', fontsize=20, 
                   color='white', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(theorems)
        ax.set_xlim(0, 1)
        ax.set_title('Theoretical Verification Status', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        
        # Add legend
        verified_patch = mpatches.Patch(color=self.colors['success'], label='Verified')
        unverified_patch = mpatches.Patch(color=self.colors['danger'], label='Not Verified')
        ax.legend(handles=[verified_patch, unverified_patch], loc='upper right')
    
    def _plot_profitability_distribution(self, ax, simulation_results: Dict):
        """Plot profitability analysis"""
        if 'profitability' not in simulation_results:
            ax.text(0.5, 0.5, 'Simulation\nResults\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        profitability = simulation_results['profitability']
        
        categories = ['Protocol', 'LP']
        positive_pcts = [
            profitability.get('protocol_profit_positive_pct', 0),
            profitability.get('lp_profit_positive_pct', 0)
        ]
        
        x = np.arange(len(categories))
        bars = ax.bar(x, positive_pcts, color=[self.colors['primary'], self.colors['secondary']], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels
        for bar, pct in zip(bars, positive_pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Profitable Scenarios (%)')
        ax.set_title('Stakeholder Profitability', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add target line
        ax.axhline(y=80, color=self.colors['success'], linestyle='--', alpha=0.7, label='Target (80%)')
        ax.legend()
    
    def _plot_risk_analysis(self, ax, simulation_results: Dict):
        """Plot risk analysis"""
        if 'risk_metrics' not in simulation_results:
            ax.text(0.5, 0.5, 'Risk Metrics\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        risk_metrics = simulation_results['risk_metrics']
        
        metrics = ['VaR 95%', 'VaR 99%', 'Max Loss']
        values = [
            risk_metrics.get('var_95_hack_losses', 0) / 1e6,
            risk_metrics.get('var_99_hack_losses', 0) / 1e6,
            risk_metrics.get('max_hack_losses', 0) / 1e6
        ]
        
        bars = ax.bar(metrics, values, color=self.colors['danger'], alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'${value:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Loss Amount ($M)')
        ax.set_title('Risk Exposure Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add insolvency risk text
        insolvency_risk = risk_metrics.get('insolvency_probability', 0)
        ax.text(0.98, 0.98, f'Insolvency Risk: {insolvency_risk:.1f}%', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8),
               fontweight='bold')
    
    def _plot_market_dynamics(self, ax, simulation_results: Dict):
        """Plot market dynamics"""
        if 'utilization' not in simulation_results:
            ax.text(0.5, 0.5, 'Market Dynamics\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        utilization = simulation_results['utilization']
        
        # Create utilization distribution visualization
        avg_util = utilization.get('avg_final_utilization', 0)
        high_util_pct = utilization.get('high_utilization_pct', 0)
        extreme_util_pct = utilization.get('extreme_utilization_pct', 0)
        
        # Create stacked bar showing utilization risk levels
        categories = ['Normal\n(<1.5)', 'High\n(1.5-3.0)', 'Extreme\n(>3.0)']
        values = [
            100 - high_util_pct,
            high_util_pct - extreme_util_pct,
            extreme_util_pct
        ]
        colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]
        
        bottom = 0
        for i, (value, color) in enumerate(zip(values, colors)):
            ax.bar('Utilization\nDistribution', value, bottom=bottom, color=color, 
                  alpha=0.8, edgecolor='black', linewidth=0.5, label=categories[i])
            
            if value > 5:  # Only show label if segment is large enough
                ax.text(0, bottom + value/2, f'{value:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white' if value > 15 else 'black')
            bottom += value
        
        ax.set_ylabel('Percentage of Scenarios')
        ax.set_title('Market Utilization Profile', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add average utilization text
        ax.text(0.5, 0.02, f'Average: {avg_util:.2f}', transform=ax.transAxes, 
               ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8))
    
    def _plot_risk_return_matrix(self, ax, simulation_results: Dict):
        """Create risk-return matrix"""
        # Create synthetic data for risk-return visualization
        scenarios = ['Conservative', 'Balanced', 'Aggressive']
        returns = [15, 25, 40]  # Expected returns
        risks = [5, 15, 30]     # Risk levels
        
        scatter = ax.scatter(risks, returns, s=[300, 400, 500], alpha=0.7,
                           c=[self.colors['success'], self.colors['warning'], self.colors['danger']])
        
        # Add labels
        for i, scenario in enumerate(scenarios):
            ax.annotate(scenario, (risks[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold')
        
        ax.set_xlabel('Risk Level (%)')
        ax.set_ylabel('Expected Return (%)')
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add efficient frontier line
        x_line = np.linspace(0, 35, 100)
        y_line = 0.8 * x_line + 10  # Simple efficient frontier
        ax.plot(x_line, y_line, '--', color=self.colors['neutral'], alpha=0.7, label='Efficient Frontier')
        ax.legend()
    
    def _create_recommendations_panel(self, ax, optimization_results: Dict, 
                                    simulation_results: Dict, theoretical_results: Dict):
        """Create recommendations panel"""
        ax.axis('off')
        
        # Generate recommendations based on results
        recommendations = self._generate_recommendations(optimization_results, simulation_results, theoretical_results)
        
        ax.text(0.5, 0.95, 'Key Recommendations', ha='center', va='top', 
               transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        for i, rec in enumerate(recommendations[:5]):  # Show top 5
            y_pos = 0.85 - i * 0.15
            ax.text(0.05, y_pos, f"{i+1}.", ha='left', va='top', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', color=self.colors['primary'])
            ax.text(0.1, y_pos, rec, ha='left', va='top', transform=ax.transAxes, 
                   fontsize=11, wrap=True)
    
    def _create_health_scorecard(self, ax, simulation_results: Dict):
        """Create market health scorecard"""
        # Calculate health scores
        health_metrics = {
            'Profitability': self._calculate_profitability_score(simulation_results),
            'Stability': self._calculate_stability_score(simulation_results),
            'Risk Management': self._calculate_risk_score(simulation_results),
            'Coverage': self._calculate_coverage_score(simulation_results),
            'Overall': 0  # Will calculate as average
        }
        
        # Calculate overall score
        health_metrics['Overall'] = np.mean(list(health_metrics.values())[:-1])
        
        # Create horizontal bar chart
        y_pos = np.arange(len(health_metrics))
        scores = list(health_metrics.values())
        
        # Color bars based on scores
        bar_colors = []
        for score in scores:
            if score >= 80:
                bar_colors.append(self.colors['success'])
            elif score >= 60:
                bar_colors.append(self.colors['warning'])
            else:
                bar_colors.append(self.colors['danger'])
        
        bars = ax.barh(y_pos, scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{score:.0f}',
                   ha='left', va='center', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(health_metrics.keys())
        ax.set_xlabel('Health Score')
        ax.set_title('Market Health Scorecard', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add reference lines
        ax.axvline(x=80, color=self.colors['success'], linestyle='--', alpha=0.7, label='Excellent')
        ax.axvline(x=60, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Good')
        ax.legend(loc='lower right')
    
    # Helper methods for calculations
    def _calculate_viability_score(self, opt_results: Dict, sim_results: Dict, theo_results: Dict) -> float:
        """Calculate overall market viability score"""
        scores = []
        
        # Optimization success (25%)
        if opt_results.get('optimal_params'):
            scores.append(85)
        else:
            scores.append(40)
        
        # Simulation profitability (35%)
        if 'profitability' in sim_results:
            protocol_prof = sim_results['profitability'].get('protocol_profit_positive_pct', 0)
            lp_prof = sim_results['profitability'].get('lp_profit_positive_pct', 0)
            scores.append((protocol_prof + lp_prof) / 2)
        else:
            scores.append(50)
        
        # Theoretical verification (25%)
        if theo_results:
            verified_count = sum([
                theo_results.get('equilibrium', {}).get('theorem_verified', False),
                theo_results.get('risk_assessment', {}).get('proposition_verified', False),
                theo_results.get('lp_dynamics', {}).get('theorem_verified', False),
                theo_results.get('solvency', {}).get('proposition_verified', False)
            ])
            scores.append(verified_count / 4 * 100)
        else:
            scores.append(50)
        
        # Risk management (15%)
        if 'risk_metrics' in sim_results:
            insolvency_risk = sim_results['risk_metrics'].get('insolvency_probability', 20)
            scores.append(max(0, 100 - insolvency_risk * 3))
        else:
            scores.append(50)
        
        return np.average(scores, weights=[0.25, 0.35, 0.25, 0.15])
    
    def _calculate_readiness_score(self, theoretical_results: Dict) -> float:
        """Calculate implementation readiness score"""
        if not theoretical_results:
            return 30
        
        verified_theorems = sum([
            theoretical_results.get('equilibrium', {}).get('theorem_verified', False),
            theoretical_results.get('risk_assessment', {}).get('proposition_verified', False),
            theoretical_results.get('lp_dynamics', {}).get('theorem_verified', False),
            theoretical_results.get('solvency', {}).get('proposition_verified', False)
        ])
        
        return (verified_theorems / 4) * 100
    
    def _calculate_profitability_score(self, simulation_results: Dict) -> float:
        """Calculate profitability health score"""
        if 'profitability' not in simulation_results:
            return 50
        
        protocol_prof = simulation_results['profitability'].get('protocol_profit_positive_pct', 0)
        lp_prof = simulation_results['profitability'].get('lp_profit_positive_pct', 0)
        return (protocol_prof + lp_prof) / 2
    
    def _calculate_stability_score(self, simulation_results: Dict) -> float:
        """Calculate stability health score"""
        if 'utilization' not in simulation_results:
            return 50
        
        stability = simulation_results['utilization'].get('utilization_stability', 0.5)
        return stability * 100
    
    def _calculate_risk_score(self, simulation_results: Dict) -> float:
        """Calculate risk management health score"""
        if 'risk_metrics' not in simulation_results:
            return 50
        
        insolvency_risk = simulation_results['risk_metrics'].get('insolvency_probability', 20)
        return max(0, 100 - insolvency_risk * 2)
    
    def _calculate_coverage_score(self, simulation_results: Dict) -> float:
        """Calculate coverage efficiency health score"""
        if 'summary_stats' not in simulation_results:
            return 70  # Default reasonable score
        
        loss_ratio = simulation_results['summary_stats'].get('avg_loss_ratio', 0.5)
        # Good loss ratio is around 0.7-0.8 (covers most losses but not over-insurance)
        if 0.6 <= loss_ratio <= 0.9:
            return 90
        elif 0.4 <= loss_ratio < 0.6 or 0.9 < loss_ratio <= 1.2:
            return 70
        else:
            return 50
    
    def _generate_recommendations(self, opt_results: Dict, sim_results: Dict, theo_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check profitability
        if 'profitability' in sim_results:
            protocol_prof = sim_results['profitability'].get('protocol_profit_positive_pct', 0)
            lp_prof = sim_results['profitability'].get('lp_profit_positive_pct', 0)
            
            if protocol_prof < 70:
                recommendations.append("Increase protocol incentives through revenue sharing adjustments")
            if lp_prof < 70:
                recommendations.append("Enhance LP risk compensation to attract more capital")
        
        # Check risk levels
        if 'risk_metrics' in sim_results:
            insolvency_risk = sim_results['risk_metrics'].get('insolvency_probability', 0)
            if insolvency_risk > 10:
                recommendations.append("Implement additional capital buffers for extreme scenarios")
        
        # Check utilization
        if 'utilization' in sim_results:
            high_util = sim_results['utilization'].get('high_utilization_pct', 0)
            if high_util > 20:
                recommendations.append("Monitor utilization levels and adjust LP incentives dynamically")
        
        # Theoretical recommendations
        if theo_results:
            if not theo_results.get('equilibrium', {}).get('theorem_verified', False):
                recommendations.append("Refine equilibrium finding mechanisms before deployment")
        
        # General recommendations
        recommendations.extend([
            "Implement gradual rollout with limited exposure initially",
            "Establish real-time monitoring dashboards for market health",
            "Create governance mechanisms for parameter updates",
            "Develop emergency procedures for extreme market conditions",
            "Consider reinsurance partnerships for tail risk management"
        ])
        
        return recommendations

def create_publication_plots(results_data: Dict, save_directory: str = "plots/"):
    """Create publication-quality plots for academic paper"""
    import os
    os.makedirs(save_directory, exist_ok=True)
    
    visualizer = ProfessionalVisualizer()
    
    # Extract data
    optimization_results = results_data.get('optimization', {})
    simulation_results = results_data.get('simulation', {})
    theoretical_results = results_data.get('theoretical', {})
    
    # Create executive dashboard
    print("📊 Creating executive dashboard...")
    fig = visualizer.create_executive_dashboard(
        optimization_results, simulation_results, theoretical_results,
        save_path=os.path.join(save_directory, "executive_dashboard.png")
    )
    
    print(f"✅ Plots saved to {save_directory}")
    return visualizer

# Example usage
if __name__ == "__main__":
    # Example data structure
    example_results = {
        'optimization': {
            'optimal_params': type('Params', (), {
                'mu': 1500, 'theta': 0.6, 'xi': 0.2, 'alpha': 0.7,
                'u_target': 0.3, 'r_pool': 0.09
            })()
        },
        'simulation': {
            'profitability': {
                'protocol_profit_positive_pct': 85,
                'lp_profit_positive_pct': 78
            },
            'risk_metrics': {
                'insolvency_probability': 5.2,
                'var_95_hack_losses': 25e6,
                'var_99_hack_losses': 45e6,
                'max_hack_losses': 85e6
            },
            'utilization': {
                'avg_final_utilization': 0.65,
                'utilization_stability': 0.85,
                'high_utilization_pct': 12,
                'extreme_utilization_pct': 3
            },
            'summary_stats': {
                'avg_loss_ratio': 0.75
            }
        },
        'theoretical': {
            'equilibrium': {'theorem_verified': True},
            'risk_assessment': {'proposition_verified': True},
            'lp_dynamics': {'theorem_verified': True},
            'solvency': {'proposition_verified': True}
        }
    }
    
    visualizer = create_publication_plots(example_results)
    print("🎨 Professional visualizations created successfully!")
