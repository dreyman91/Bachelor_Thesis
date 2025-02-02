from Market_Vendors_and_Customers.market_aec import Market_AEC_Env

env = Market_AEC_Env(customers=3, vendors=3, admin=1)
env.reset()
print("Environment initiated succesfully")

