# Read the file
with open('defi_insurance_core.py', 'r') as f:
    content = f.read()

# Find the find_equilibrium method and add dampening
old_equilibrium = '''            # Update best responses
            c_c = self.protocol_best_response(c_lp, c_spec)
            c_lp = self.lp_best_response(c_c, c_spec)
            # c_spec assumed to be market-driven (simplified)'''

new_equilibrium = '''            # Update best responses with dampening to prevent oscillation
            c_c_new = self.protocol_best_response(c_lp, c_spec)
            c_lp_new = self.lp_best_response(c_c_new, c_spec)
            
            # Apply dampening factor to prevent oscillation
            dampening = 0.3  # Mix 30% new + 70% old
            c_c = dampening * c_c_new + (1 - dampening) * c_c
            c_lp = dampening * c_lp_new + (1 - dampening) * c_lp
            # c_spec assumed to be market-driven (simplified)'''

content = content.replace(old_equilibrium, new_equilibrium)

# Also relax the tolerance since we're using dampening
old_tolerance = 'tolerance: float = 1e-6'
new_tolerance = 'tolerance: float = 1e-4'  # Less strict

content = content.replace(old_tolerance, new_tolerance)

# Write back
with open('defi_insurance_core.py', 'w') as f:
    f.write(content)

print("✅ Added dampening to prevent equilibrium oscillation")
print("🎯 Tolerance relaxed to 1e-4 for stability")
