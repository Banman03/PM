from py_clob_client.client import ClobClient

# host: str = ""
# key: str = ""
# chain_id: int = 137

# ### Initialization of a client that trades directly from an EOA
# client = ClobClient(host, key=key, chain_id=chain_id)

# ### Initialization of a client using a Polymarket Proxy associated with an Email/Magic account
# client = ClobClient(host, key=key, chain_id=chain_id, signature_type=1, funder=POLYMARKET_PROXY_ADDRESS)

# ### Initialization of a client using a Polymarket Proxy associated with a Browser Wallet(Metamask, Coinbase Wallet, etc)
# client = ClobClient(host, key=key, chain_id=chain_id, signature_type=2, funder=POLYMARKET_PROXY_ADDRESS)


client = ClobClient("https://clob.polymarket.com")  # Level 0 (no auth)

ok = client.get_ok()
time = client.get_server_time()
print(ok, time)