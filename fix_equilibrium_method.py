# Read the file
with open('defi_insurance_core.py', 'r') as f:
    content = f.read()

# Find and replace the calls in protocol_best_response
old_protocol_call = 'profit = self.market.protocol_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma)'
new_protocol_call = 'profit = self.market.protocol_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma, risk_aversion=2.0)'

content = content.replace(old_protocol_call, new_protocol_call)

# Find and replace the calls in lp_best_response  
old_lp_call = 'profit = self.market.lp_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma)'
new_lp_call = 'profit = self.market.lp_profit(c_c, c_lp, c_spec, p_hack, expected_lgh, gamma, risk_compensation=2.0)'

content = content.replace(old_lp_call, new_lp_call)

# Write back
with open('defi_insurance_core.py', 'w') as f:
    f.write(content)

print("✅ Fixed EquilibriumSolver to use risk aversion parameters")
print("🔧 Both best response functions now use fixed profit calculations")
