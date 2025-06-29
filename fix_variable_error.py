with open('theoretical_proofs.py', 'r') as f:
    content = f.read()

# Find and fix the variable name error
old_final_return = '''        # Check convergence to behavioral equilibrium
        final_lp = c_lp_path[-1]
        convergence_error = abs(final_lp - equilibrium_c_lp) / equilibrium_c_lp
        
        print(f"  Final LP return: {final_return:.4f}")
        print(f"  Target return: {target_return:.4f}")
        print(f"  Convergence error: {convergence_error:.6f}")'''

new_final_return = '''        # Check convergence to behavioral equilibrium
        final_lp = c_lp_path[-1]
        convergence_error = abs(final_lp - equilibrium_c_lp) / equilibrium_c_lp
        
        # Calculate final return for reporting
        final_coverage = self.market.coverage_function(c_c, self.market.tvl)
        final_u = final_coverage / final_lp if final_lp > 0 else float('inf')
        final_gamma = self.market.revenue_share_function(final_u, self.params.p_baseline)
        final_profit = self.market.lp_profit(c_c, final_lp, c_spec, p_hack, expected_lgh, final_gamma, risk_compensation=2.0)
        final_return = final_profit / final_lp if final_lp > 0 else 0
        
        print(f"  Final LP return: {final_return:.4f}")
        print(f"  Target return: {target_return:.4f}")
        print(f"  Convergence error: {convergence_error:.6f}")'''

content = content.replace(old_final_return, new_final_return)

with open('theoretical_proofs.py', 'w') as f:
    f.write(content)

print("✅ Fixed variable name error in self-stabilization test")
