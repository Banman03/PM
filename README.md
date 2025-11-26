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

- Implementation plan below:

'''
  Data Requirements & CLOB Endpoints

  1. Trading Intensity I(τ) - Trade Volume Data

  Endpoint:
  GET https://clob.polymarket.com/trades?market={condition_id}

  What to collect:
  - Trade size, price, timestamp for each trade
  - Aggregate over time intervals to calculate volume per unit time
  - Can also use your WebSocket mode (0) to capture real-time trades

  How to use:
  - Poll this endpoint periodically (e.g., every 1-5 seconds)
  - Calculate I(τ) as: sum(trade_sizes) / time_interval
  - Higher trading volume means more liquidity depletion

  ---
  2. Bid-Ask Spread R(τ)

  Endpoint:
  GET https://clob.polymarket.com/book?token_id={token_id}

  What to collect:
  - Best bid price (highest buy order)
  - Best ask price (lowest sell order)
  - Calculate spread: R(τ) = best_ask - best_bid

  You already have this! Your REST API mode (1) fetches the order book, which contains:
  {
    "bids": [[price, size], ...],
    "asks": [[price, size], ...]
  }

  The spread is: asks[0][0] - bids[0][0]

  ---
  3. Liquidity Level ℓ(τ)

  Endpoint: Same as above
  GET https://clob.polymarket.com/book?token_id={token_id}

  What to calculate:
  Several ways to measure liquidity from the order book:

  Option A - Depth at best prices:
  liquidity = bids[0][1] + asks[0][1]  # size at best bid + size at best ask

  Option B - Total order book depth (better):
  total_bid_volume = sum(size for price, size in bids)
  total_ask_volume = sum(size for price, size in asks)
  liquidity = total_bid_volume + total_ask_volume

  Option C - Depth within spread % (most realistic):
  # Sum all orders within 1% of mid-price
  mid_price = (best_bid + best_ask) / 2
  liquidity = sum(size for p, size in bids if p >= mid_price * 0.99) + \
              sum(size for p, size in asks if p <= mid_price * 1.01)

  ---
  Recommended Data Collection Strategy

  Hybrid Approach:

  1. Use REST API mode (1) for order book snapshots (already implemented)
    - Poll every 5-10 seconds
    - Extract: bid-ask spread R(τ), liquidity level ℓ(τ)
  2. Add trades endpoint polling (new implementation needed)
    - Poll trades every 1-5 seconds
    - Calculate: trading intensity I(τ)
  3. Store time-series data:
  {
      "timestamp": 1234567890,
      "l_t": 1500.5,           # liquidity level
      "R_t": -0.002,           # bid-ask spread (negative)
      "I_t": 45.3,             # trading intensity
      "dl_dt": -2.1            # rate of change (calculate from consecutive measurements)
  }

  ---
  Implementation Suggestions

  Extend your REST API poller to include trades:

  def get_trades(self, market_id: str, since_timestamp: int = None):
      """Fetch recent trades for a market"""
      url = f"{self.CLOB_API}/trades"
      params = {"market": market_id}
      if since_timestamp:
          params["after"] = since_timestamp

      response = requests.get(url, params=params)
      return response.json()

  def calculate_trading_intensity(self, trades: List, time_window: int = 60):
      """Calculate volume per unit time"""
      total_volume = sum(float(t["size"]) for t in trades)
      return total_volume / time_window

  Calculate the differential equation terms:

  def calculate_liquidity_dynamics(current_data, previous_data, dt):
      """Calculate dℓ/dτ and the three terms in the equation"""

      # Rate of change
      dl_dt = (current_data["l_t"] - previous_data["l_t"]) / dt

      # Three terms from equation (6)
      term1 = -a * current_data["I_t"]              # Trading depletion
      term2 = -b * current_data["R_t"]              # Spread inflow
      term3 = c * current_data["l_t"]               # Natural decay

      return {
          "dl_dt_observed": dl_dt,
          "dl_dt_predicted": term1 + term2 + term3,
          "trading_term": term1,
          "spread_term": term2,
          "decay_term": term3
      }

  1. Trades endpoint polling
  2. Calculate I(τ) from trades
  3. Calculate dℓ/dτ from consecutive liquidity measurements
  4. Fit the model to estimate coefficients a, b, c
'''