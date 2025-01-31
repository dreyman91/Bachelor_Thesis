from Market_Vendors_and_Customers.Market_AEC import Market_AEC_Env

env = Market_AEC_Env(customers=3, vendors=3, admin=1)

obs = env.reset()
print(f"Initial Observation: {obs}")

action = {"vendor_0": "add_product", "customer_0": "buy_product"}
obs, reward, terminations, truncations, info = env.step(action)

print(f"Observation: {obs}")
print(f"Reward: {reward}")
print(f"Termination: {terminations}")
print(f"Truncations: {truncations}")
print(f"Info: {info}")