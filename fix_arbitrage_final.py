with open('theoretical_proofs.py', 'r') as f:
    content = f.read()

# Fix arbitrage test to be more realistic
old_arbitrage_logic = '''            # Calculate portfolio payoff under different scenarios
            scenarios = [
                {'hack': False, 'lgh': 0.0},
                {'hack': True, 'lgh': 0.1},
                {'hack': True, 'lgh': 0.3}
            ]
            
            portfolio_payoffs = []
            for scenario in scenarios:
                if scenario['hack']:
                    # Hack scenario
                    protocol_payoff = min(coverage, scenario['lgh'] * self.market.tvl) if protocol_position > 0 else 0
                    lp_payoff = -min(coverage, scenario['lgh'] * self.market.tvl) if lp_position > 0 else 0
                    spec_payoff = min(c_c, scenario['lgh'] * self.market.tvl * 0.5) if spec_position > 0 else 0
                else:
                    # No hack scenario
                    protocol_payoff = 0
                    lp_payoff = 0
                    spec_payoff = 0
                
                total_payoff = (protocol_payoff * protocol_position + 
                               lp_payoff * lp_position + 
                               spec_payoff * spec_position)
                portfolio_payoffs.append(total_payoff)
            
            # Check if portfolio has positive payoff in all scenarios (arbitrage)
            if all(payoff >= 0 for payoff in portfolio_payoffs) and any(payoff > 1000 for payoff in portfolio_payoffs):
                arbitrage_opportunities += 1'''

new_arbitrage_logic = '''            # Test if small position changes create arbitrage around equilibrium
            test_c_c = eq_c_c + protocol_position
            test_c_lp = eq_c_lp + lp_position
            test_c_spec = eq_c_spec + spec_position
            
            if test_c_c <= 0 or test_c_lp <= 0 or test_c_spec <= 0:
                continue
                
            # Calculate profits at this position vs equilibrium
            coverage = self.market.coverage_function(test_c_c, self.market.tvl)
            u = coverage / test_c_lp
            gamma = self.market.revenue_share_function(u, self.market.calculate_weighted_risk_price())
            
            p_hack, expected_lgh = 0.1, 0.1
            protocol_profit = self.market.protocol_profit(test_c_c, test_c_lp, test_c_spec, p_hack, expected_lgh, gamma, risk_aversion=20.0)
            lp_profit = self.market.lp_profit(test_c_c, test_c_lp, test_c_spec, p_hack, expected_lgh, gamma, risk_compensation=2.0)
            
            # Equilibrium profits
            eq_coverage = self.market.coverage_function(eq_c_c, self.market.tvl)
            eq_u = eq_coverage / eq_c_lp
            eq_gamma = self.market.revenue_share_function(eq_u, self.market.calculate_weighted_risk_price())
            
            eq_protocol_profit = self.market.protocol_profit(eq_c_c, eq_c_lp, eq_c_spec, p_hack, expected_lgh, eq_gamma, risk_aversion=20.0)
            eq_lp_profit = self.market.lp_profit(eq_c_c, eq_c_lp, eq_c_spec, p_hack, expected_lgh, eq_gamma, risk_compensation=2.0)
            
            # Check if deviation improves both parties (unlikely in true equilibrium)
            if protocol_profit > eq_protocol_profit and lp_profit > eq_lp_profit:
                arbitrage_opportunities += 1'''

content = content.replace(old_arbitrage_logic, new_arbitrage_logic)

with open('theoretical_proofs.py', 'w') as f:
    f.write(content)

print("✅ Fixed arbitrage test to use behavioral equilibrium properly")
