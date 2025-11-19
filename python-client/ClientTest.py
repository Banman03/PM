from py_clob_client.client import ClobClient
import os

HOST = os.environ['HOST']
print(HOST)
CHAIN_ID = 137
PRIVATE_KEY = "<your-private-key>"
PROXY_FUNDER = "<your-proxy-or-smart-wallet-address>"  # Address that holds your funds

client = ClobClient(
    HOST,  # The CLOB API endpoint
    key=PRIVATE_KEY,  # Your wallet's private key
    chain_id=CHAIN_ID,  # Polygon chain ID (137)
    signature_type=1,  # 1 for email/Magic wallet signatures
    funder=PROXY_FUNDER  # Address that holds your funds
)
client.set_api_creds(client.create_or_derive_api_creds())