"""
DeFi Cybersecurity Insurance Market Mechanism - Core Model
Implementation of the mathematical framework from the paper
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MarketParameters:
    """Core market parameters for the insurance mechanism"""
    # Coverage function parameters
    mu: float = 2.0  # Coverage amplification factor
    theta: float = 0.7  # Coverage concavity parameter (0, 1)
    xi: float = 0.2  # Security scaling factor (0, 1)
    
    # Revenue share function parameters
    alpha: float = 0.6  # Weight on utilization in gamma function
    beta: float = 1.5  # Convexity parameter for utilization
    delta: float = 1.2  # Convexity parameter for risk pricing
    
    # Target utilization
    u_target: float = 0.8
    
    # Market parameters
    r_market: float = 0.05  # Opportunity return rate
    r_pool: float = 0.06  # Insurance pool yield rate
    rho: float = 0.02  # Risk premium for LPs
    
    # Hack probability parameters
    lambda_hack: float = 0.1  # Annual hack intensity (Poisson)
    p_baseline: float = 0.01  # Baseline LGH price
    
    # Premium rate
    premium_rate: float = 0.01  # Annual premium as % of TVL


class InsuranceMarket:
    """Main class implementing the DeFi insurance market mechanism"""
    
    def __init__(self, params: MarketParameters):
        self.params = params
        self.reset_state()
    
    def reset_state(self):
        """Reset market state to initial conditions"""
        self.c_c = 0.0  # Protocol collateral
        self.c_lp = 0.0  # LP capital
        self.c_spec = 0.0  # Speculator capital
        self.tvl = 100_000_000.0  # Default TVL: $100M
        self.lgh_prices = {}  # LGH token prices by strike
        self.lgh_weights = {}  # LGH token weights by strike
    
    def coverage_function(self, c_c: float, tvl: float, security_factor: float = 1.0) -> float:
        """
        Calculate maximum coverage based on collateral (Equation 2)
        
        Args:
            c_c: Protocol collateral
            tvl: Total value locked
            security_factor: (1 + xi) where xi reflects security measures
        
        Returns:
            Maximum coverage amount
        """
        return self.params.mu * (c_c ** self.params.theta) * security_factor
    
    def utilization(self, coverage: float, c_lp: float) -> float:
        """
        Calculate utilization ratio (Equation 3)
        
        Args:
            coverage: Maximum coverage amount
            c_lp: LP capital
        
        Returns:
            Utilization ratio
        """
        if c_lp <= 0:
            return float('inf')
        return coverage / c_lp
    
    def revenue_share_function(self, u: float, p_risk: float) -> float:
        """
        Calculate LP revenue share gamma (Equation 4)
        
        Args:
            u: Utilization ratio
            p_risk: Weighted average LGH token price
        
        Returns:
            Revenue share for LPs (bounded between 0 and 1)
        """
        u_term = self.params.alpha * (u / self.params.u_target) ** self.params.beta
        p_term = (1 - self.params.alpha) * (p_risk / self.params.p_baseline) ** self.params.delta
        
        gamma = u_term + p_term
        return max(0.0, min(1.0, gamma))
    
    def total_capital(self, c_c: float, c_premium: float, c_lp: float, c_spec: float) -> float:
        """Calculate total insurance capital pool (Equation 1)"""
        return c_c + c_premium + c_lp + c_spec
    
    def protocol_profit(self, c_c: float, c_lp: float, c_spec: float, 
                p_hack: float, expected_lgh: float, gamma: float, risk_aversion: float = 2.0) -> float:
        """
        Calculate protocol expected profit (Equation 6) with proper risk aversion scaling
        
        Args:
            c_c: Protocol collateral
            c_lp: LP capital
            c_spec: Speculator capital
            p_hack: Probability of hack
            expected_lgh: Expected LGH value
            gamma: Revenue share for LPs
            risk_aversion: Risk aversion coefficient (should be much higher for DeFi protocols)
        
        Returns:
            Expected protocol profit with risk aversion utility
        """
        # Calculate components
        coverage = self.coverage_function(c_c, self.tvl)
        c_premium = self.params.premium_rate * self.tvl
        total_cap = self.total_capital(c_c, c_premium, c_lp, c_spec)
        
        # Expected insurance payout
        expected_loss = expected_lgh * self.tvl
        insurance_payout = p_hack * min(coverage, expected_loss)
        
        # Speculator payout from collateral
        speculator_payout = min(c_c, self._calculate_speculator_payout(expected_lgh))
        
        # Protocol's share of pool yield
        yield_share = (1 - gamma) * self.params.r_pool * total_cap
        
        # Opportunity cost and premium
        opportunity_cost = c_c * self.params.r_market
        premium_cost = c_premium
        
        # FIXED: Proper risk aversion utility calculation
        # Risk aversion should reflect the enormous reputational/operational cost of user fund loss
        expected_loss_amount = expected_lgh * self.tvl
        uninsured_loss_risk = p_hack * expected_loss_amount
        coverage_amount = self.coverage_function(c_c, self.tvl)
        insured_loss_risk = p_hack * max(0, expected_loss_amount - coverage_amount)
        
        # Risk reduction value should be much larger for protocols managing user funds
        # DeFi protocols face existential risk from hacks - users lose trust and abandon protocol
        risk_reduction_value = (uninsured_loss_risk - insured_loss_risk) * risk_aversion
        
        # FIXED: For DeFi protocols, risk aversion should be 100-1000x higher
        # Losing user funds = protocol death, not just monetary loss
        if risk_aversion < 50:  # If using low risk aversion, scale it up for DeFi reality
            risk_aversion_scaled = risk_aversion * 50  # Scale up to realistic DeFi levels
            risk_reduction_value = (uninsured_loss_risk - insured_loss_risk) * risk_aversion_scaled

        # Basic monetary profit/loss
        basic_profit = insurance_payout - speculator_payout + yield_share - opportunity_cost - premium_cost
        
        # Total utility = monetary profit + risk reduction utility
        total_utility = basic_profit + risk_reduction_value
        
        return total_utility

    def lp_profit(self, c_c: float, c_lp: float, c_spec: float,
                  p_hack: float, expected_lgh: float, gamma: float, risk_compensation: float = 1.5) -> float:
        """
        Calculate LP expected profit (Equation 8)
        
        Args:
            c_c: Protocol collateral
            c_lp: LP capital
            c_spec: Speculator capital
            p_hack: Probability of hack
            expected_lgh: Expected LGH value
            gamma: Revenue share for LPs
        
        Returns:
            Expected LP profit
        """
        # Calculate components
        coverage = self.coverage_function(c_c, self.tvl)
        c_premium = self.params.premium_rate * self.tvl
        total_cap = self.total_capital(c_c, c_premium, c_lp, c_spec)
        
        # LP's share of pool yield
        yield_share = gamma * self.params.r_pool * total_cap
        
        # Expected payout obligation
        expected_loss = expected_lgh * self.tvl
        expected_payout = p_hack * min(coverage, expected_loss)
        
        # Opportunity cost
        opportunity_cost = c_lp * self.params.r_market
        
        # Add risk compensation for capital at risk
        # LPs should be compensated for the risk they bear beyond just yield
        coverage = self.coverage_function(c_c, self.tvl)
        capital_at_risk = min(c_lp, coverage)  # Amount that could be lost
        risk_compensation_value = capital_at_risk * self.params.rho * risk_compensation
        
        return yield_share - expected_payout - opportunity_cost + risk_compensation_value
    
    def lgh_token_value(self, strike_l: float, time_horizon: float, p_hack: float,
                       p_lgh_given_hack: float, expected_payout: float) -> float:
        """
        Calculate fair value of LGH token (Equation 9)
        
        Args:
            strike_l: LGH strike level
            time_horizon: Time to expiry
            p_hack: Probability of hack in time horizon
            p_lgh_given_hack: P(LGH >= L | hack)
            expected_payout: Expected payout if LGH >= L
        
        Returns:
            Fair value of LGH token
        """
        discount_factor = np.exp(-self.params.r_market * time_horizon)
        return p_hack * p_lgh_given_hack * expected_payout * discount_factor
    
    def _calculate_speculator_payout(self, realized_lgh: float) -> float:
        """Calculate total speculator payout based on realized LGH"""
        total_payout = 0.0
        
        for strike, weight in self.lgh_weights.items():
            if realized_lgh >= strike:
                # Simplified payout calculation
                payout_per_token = min(1.0, strike / max(realized_lgh, 0.001))
                total_payout += weight * payout_per_token
        
        return total_payout
    
    def calculate_weighted_risk_price(self) -> float:
        """Calculate weighted average LGH token price"""
        if not self.lgh_prices or not self.lgh_weights:
            return self.params.p_baseline
        
        total_weight = sum(self.lgh_weights.values())
        if total_weight == 0:
            return self.params.p_baseline
        
        weighted_price = sum(
            price * self.lgh_weights.get(strike, 0) 
            for strike, price in self.lgh_prices.items()
        )
        
        return weighted_price / total_weight
    
    def update_lgh_prices(self, prices: Dict[float, float], weights: Dict[float, float]):
        """Update LGH token prices and weights"""
        self.lgh_prices = prices.copy()
        self.lgh_weights = weights.copy()
    
    def simulate_hack_probability(self, time_horizon: float = 1.0) -> float:
        """Simulate hack probability using Poisson process"""
        return 1 - np.exp(-self.params.lambda_hack * time_horizon)
    
    def get_market_state(self) -> Dict:
        """Get current market state summary"""
        coverage = self.coverage_function(self.c_c, self.tvl)
        u = self.utilization(coverage, self.c_lp) if self.c_lp > 0 else float('inf')
        p_risk = self.calculate_weighted_risk_price()
        gamma = self.revenue_share_function(u, p_risk)
        
        return {
            'collateral': self.c_c,
            'lp_capital': self.c_lp,
            'speculator_capital': self.c_spec,
            'tvl': self.tvl,
            'coverage': coverage,
            'utilization': u,
            'risk_price': p_risk,
            'revenue_share': gamma,
            'total_capital': self.total_capital(
                self.c_c, 
                self.params.premium_rate * self.tvl,
                self.c_lp, 
                self.c_spec
            )
        }


class EquilibriumSolver:
    """Solver for finding Nash equilibrium of the three-party game"""
    
    def __init__(self, market: InsuranceMarket):
        self.market = market
    
    def protocol_best_response(self, c_lp: float, c_spec: float, 
                             max_collateral: float = None) -> float:
        """Find protocol's best response collateral amount"""
        if max_collateral is None:
            max_collateral = self.market.tvl * 0.5  # Max 50% of TVL
        
        def objective(c_c):
            p_hack = self.market.simulate_hack_probability()
            expected_lgh = 0.1  # Simplified assumption
            p_risk = self.market.calculate_weighted_risk_price()
            coverage = self.market.coverage_function(c_c, self.market.tvl)
            u = self.market.utilization(coverage, c_lp) if c_lp > 0 else float('inf')
            gamma = self.market.revenue_share_function(u, p_risk)
            
            return -self.market.protocol_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma)
        
        # Grid search for simplicity
        c_c_range = np.linspace(0, max_collateral, 100)
        profits = [objective(c_c) for c_c in c_c_range]
        best_idx = np.argmin(profits)
        
        return c_c_range[best_idx]
    
    def lp_best_response(self, c_c: float, c_spec: float, 
                        max_capital: float = None) -> float:
        """Find LP's best response capital amount"""
        if max_capital is None:
            max_capital = self.market.tvl * 2.0  # Max 200% of TVL
        
        def objective(c_lp):
            if c_lp <= 0:
                return float('inf')
            
            p_hack = self.market.simulate_hack_probability()
            expected_lgh = 0.1  # Simplified assumption
            p_risk = self.market.calculate_weighted_risk_price()
            coverage = self.market.coverage_function(c_c, self.market.tvl)
            u = self.market.utilization(coverage, c_lp)
            gamma = self.market.revenue_share_function(u, p_risk)
            
            return -self.market.lp_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma)
        
        # Grid search for simplicity
        c_lp_range = np.linspace(1000, max_capital, 100)  # Minimum positive capital
        profits = [objective(c_lp) for c_lp in c_lp_range]
        best_idx = np.argmin(profits)
        
        return c_lp_range[best_idx]
    
    def find_equilibrium(self, max_iterations: int = 100, 
                        tolerance: float = 1e-4) -> Tuple[float, float, float]:
        """
        Find Nash equilibrium using iterative best response
        
        Returns:
            Tuple of (optimal_c_c, optimal_c_lp, optimal_c_spec)
        """
        # Initial guesses
        c_c = self.market.tvl * 0.1
        c_lp = self.market.tvl * 0.5
        c_spec = self.market.tvl * 0.05
        
        for iteration in range(max_iterations):
            c_c_old, c_lp_old, c_spec_old = c_c, c_lp, c_spec
            
            # Update best responses with dampening to prevent oscillation
            c_c_new = self.protocol_best_response(c_lp, c_spec)
            c_lp_new = self.lp_best_response(c_c_new, c_spec)
            
            # Apply dampening factor to prevent oscillation
            dampening = 0.3  # Mix 30% new + 70% old
            c_c = dampening * c_c_new + (1 - dampening) * c_c
            c_lp = dampening * c_lp_new + (1 - dampening) * c_lp
            # c_spec assumed to be market-driven (simplified)
            
            # Check convergence
            if (abs(c_c - c_c_old) < tolerance and 
                abs(c_lp - c_lp_old) < tolerance):
                print(f"Equilibrium found after {iteration + 1} iterations")
                break
        else:
            print(f"Maximum iterations ({max_iterations}) reached")
        
        return c_c, c_lp, c_spec


if __name__ == "__main__":
    # Example usage
    params = MarketParameters()
    market = InsuranceMarket(params)
    
    # Set initial market state
    market.c_c = 10_000_000  # $10M collateral
    market.c_lp = 50_000_000  # $50M LP capital
    market.c_spec = 2_000_000  # $2M speculator capital
    
    # Update LGH prices
    lgh_prices = {0.05: 0.008, 0.10: 0.015, 0.20: 0.025}
    lgh_weights = {0.05: 1000, 0.10: 800, 0.20: 500}
    market.update_lgh_prices(lgh_prices, lgh_weights)
    
    # Get market state
    state = market.get_market_state()
    print("Market State:")
    for key, value in state.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    # Find equilibrium
    solver = EquilibriumSolver(market)
    eq_c_c, eq_c_lp, eq_c_spec = solver.find_equilibrium()
    
    print(f"\nEquilibrium:")
    print(f"Protocol collateral: ${eq_c_c:,.2f}")
    print(f"LP capital: ${eq_c_lp:,.2f}")
    print(f"Speculator capital: ${eq_c_spec:,.2f}")
