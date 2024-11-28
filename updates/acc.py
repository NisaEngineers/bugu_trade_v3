import MetaTrader5 as mt5

mt5.initialize(login=185976750, server="Exness-MT5Real26", password="*Jaman@1310#")
# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    mt5.shutdown()
else:
    print("Connected to MetaTrader 5")

# Get account information
account_info = mt5.account_info()

if account_info is None:
    print("Failed to get account information, error code =", mt5.last_error())
else:
    print("Account information:", account_info)
    print("Balance:", account_info.balance)
    print("Equity:", account_info.equity)
    print("Margin:", account_info.margin)
    print("Free Margin:", account_info.margin_free)
    print("Leverage:", account_info.leverage)

# Get open orders
orders = mt5.orders_get()
if orders is None:
    print("No open orders, error code =", mt5.last_error())
else:
    print("Total open orders:", len(orders))
    for order in orders:
        print(order)

# Get open positions
positions = mt5.positions_get()
if positions is None:
    print("No open positions, error code =", mt5.last_error())
else:
    print("Total open positions:", len(positions))
    for position in positions:
        print(position)

# Shutdown the connection
mt5.shutdown()
