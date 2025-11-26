# PM
Exploration of profit opportunities in Polymarket.

# Tooling
This project uses `https://github.com/stephenberry/glaze` and `https://github.com/boostorg/boost`.

`Boost::beast` is responsible for the wss connection to Polymarket's websocket endpoint.
`Glaze is responsible for all JSON operations.`

TODO (for 421):
- Need to access trades data `GET /<clob-endpoint>/data/trades` in order to collect information on trading intensity
    - Will poll this periodically.
- Already implemented bid-ask spread data collection via `url = f"{self.CLOB_API}/book"`
- liquidity is a function of t, so we will perchance solve this numerically.

- The real goal here is to use data in order to get approximations for constants, or to have an idea of the shape of the liquidity curve

- I will,  next, begin working on the numerical solution to the original liquidity equation I provided (see math)