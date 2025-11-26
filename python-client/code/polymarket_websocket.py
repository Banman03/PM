#!/usr/bin/env python3
"""
Polymarket WebSocket Client for Streaming Order Data from Markets

This script connects to the Polymarket CLOB WebSocket endpoint and streams
order book updates and trade data from markets.
"""

import json
from websocket import WebSocketApp
import requests
from typing import List, Dict, Any
import threading
import time
import os
import argparse

MARKET_CHANNEL = "market"
USER_CHANNEL = "user"

# Data collection modes
MODE_WEBSOCKET = 0
MODE_REST_API = 1

class PolymarketWebSocketClient:
    """Client for streaming data from Polymarket WebSocket API"""

    WS_HOST = "wss://ws-subscriptions-clob.polymarket.com"
    API_URL = "https://gamma-api.polymarket.com/markets"
    PRICE_CHANGE = "price_change"
    BOOK = "book"

    def __init__(self, asset_ids: List[str], channel: str = MARKET_CHANNEL, ws_host: str = None):
        """
        Initialize the WebSocket client

        Args:
            asset_ids: List of token IDs to subscribe to
            channel: WebSocket channel type ("market" or "user")
            ws_host: The WebSocket host (default: production CLOB WS)
        """
        self.ws_host = ws_host or self.WS_HOST
        self.channel = channel
        self.asset_ids = asset_ids
        self.message_count = 0
        self.file_dict = {}

        ws_url = f"{self.ws_host}/ws/{channel}"
        self.ws = WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        
    def write_data(self, data):
        if not isinstance(data, Dict):
            return

        event_type = data["event_type"]
            
        if event_type == self.PRICE_CHANGE:
            # print("here")
            changes = data["price_changes"]
            print(changes)
            for change in changes:
                asset_id = change["asset_id"]

                if asset_id in self.file_dict:
                    file = self.file_dict[asset_id]

                    write_payload = {
                        "timestamp": data["timestamp"],
                        "event_type": event_type,
                        "data": change
                    }

                    file.write(json.dumps(write_payload) + "\n")
                    file.flush()

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        self.message_count += 1
        try:
            data = json.loads(message)
            self.write_data(data)

            print(f"\n[Message {self.message_count}]")
            print(json.dumps(data, indent=2))
            print("-" * 80)
        except json.JSONDecodeError:
            print(f"Non-JSON message: {message}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        print(f"\nWebSocket closed - Code: {close_status_code}, Message: {close_msg}")

    def create_or_append_asset_files(self):
        for id in self.asset_ids:
            if (file_exists(id) is not None):
                file = open(file_exists(id), "a")
            else:
                base_path = "../data/streamed-data/"+id
                file = open(base_path, "w")
            self.file_dict[id] = file

    def on_open(self, ws):
        """Handle WebSocket connection open and send subscription message"""
        print(f"blished to {self.ws_host}/ws/{self.channel}")

        # Send subscription message
        subscription_message = {
            "assets_ids": self.asset_ids,
            "type": self.channel
        }

        print(f"\nSubscribing to {len(self.asset_ids)} asset(s)...")
        print(f"Subscription message: {json.dumps(subscription_message, indent=2)}\n")

        self.create_or_append_asset_files()

        ws.send(json.dumps(subscription_message))

        ping_thread = threading.Thread(target=self.ping, args=(ws,), daemon=True)
        ping_thread.start()

    def ping(self, ws):
        """Send periodic ping to keep connection alive"""
        while True:
            time.sleep(10)
            try:
                ws.send("PING")
            except Exception as e:
                print(f"Ping failed: {e}")
                break

    def run(self):
        """Start the WebSocket connection"""
        print(f"Starting WebSocket client...\n")
        self.ws.run_forever()


def file_exists(filename: str):
    base_path = "../data/streamed-data/"
    if os.path.exists(base_path+filename):
        return base_path+filename
    else:
        return None

def get_markets(offset: int = 0, closed: bool = None,
                volume_min: int = 1_000_000, liquidity_min: int = 1_000_000) -> List[Dict[str, Any]]:
    """
    Fetch markets from Polymarket API using requests

    Args:
        offset: Pagination offset (default: 0)
        closed: Filter by closed status (None = all, True = closed only, False = open only)
        volume_min: Minimum volume filter
        liquidity_min: Minimum liquidity filter

    Returns:
        List of market dictionaries with full metadata
    """
    api_url = "https://gamma-api.polymarket.com/markets"
    print(f"Fetching markets from {api_url}...")

    params = {
        "offset": offset,
        "volume_num_min": volume_min,
        "liquidity_num_min": liquidity_min
    }

    if closed is not None:
        params["closed"] = str(closed).lower()

    try:
        response = requests.get(api_url, params=params)

        if response.status_code != 200:
            print(f"ERROR: API returned status {response.status_code}")
            return []

        markets = response.json()

        json_data = json.dumps(markets, indent=2)
        with open("market-data.json", "w") as file:
            file.write(json_data)

        print(f"Fetched {len(markets)} markets")
        return markets

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        return []


class OrderBookPoller:
    """REST API poller for order book depth data"""

    CLOB_API = "https://clob.polymarket.com"

    def __init__(self, asset_ids: List[str], poll_interval: int = 10):
        """
        Initialize the order book poller

        Args:
            asset_ids: List of token IDs to poll
            poll_interval: Seconds between polls (default: 10)
        """
        self.asset_ids = asset_ids
        self.poll_interval = poll_interval
        self.running = True
        self.file_dict = {}

    def create_asset_files(self):
        """Create data files for each asset"""
        os.makedirs("data/order-books", exist_ok=True)

        for asset_id in self.asset_ids:
            file_path = f"data/order-books/{asset_id}.jsonl"
            self.file_dict[asset_id] = open(file_path, "w")
            print(f"Created file for asset {asset_id}")

    def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Fetch order book for a specific token

        Args:
            token_id: The token ID to fetch order book for

        Returns:
            Order book data with bids and asks
        """
        url = f"{self.CLOB_API}/book"
        params = {"token_id": token_id}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"ERROR: Failed to fetch order book for {token_id}: {response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed for {token_id}: {e}")
            return {}

    def poll_order_books(self):
        """Continuously poll order books and write to files"""
        print(f"\nStarting order book polling every {self.poll_interval} seconds...")
        print("Press Ctrl+C to stop\n")

        self.create_asset_files()
        poll_count = 0

        try:
            while self.running:
                poll_count += 1
                timestamp = int(time.time() * 1000)

                print(f"[Poll #{poll_count}] Fetching order books at {timestamp}...")

                for asset_id in self.asset_ids:
                    order_book = self.get_order_book(asset_id)

                    if order_book:
                        write_payload = {
                            "timestamp": timestamp,
                            "asset_id": asset_id,
                            "data": order_book
                        }

                        file = self.file_dict[asset_id]
                        file.write(json.dumps(write_payload) + "\n")
                        file.flush()

                        # Print summary
                        bids = order_book.get("bids", [])
                        asks = order_book.get("asks", [])
                        print(f"  {asset_id}: {len(bids)} bids, {len(asks)} asks")

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n\nPolling interrupted by user.")
        finally:
            self.close_files()

    def close_files(self):
        """Close all open files"""
        for file in self.file_dict.values():
            file.close()
        print("All files closed.")


def main():
    """Main function to run the data collector"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Polymarket data collector - WebSocket or REST API mode"
    )
    parser.add_argument(
        "mode",
        type=int,
        choices=[0, 1],
        help="Data collection mode: 0=WebSocket (price changes), 1=REST API (order book depth)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Polling interval in seconds for REST API mode (default: 10)"
    )

    args = parser.parse_args()

    try:
        print("=" * 80)
        print("Step 1: Fetching markets from Polymarket API")
        print("=" * 80)

        markets = get_markets(closed=False)

        if not markets:
            print("No markets found. Exiting...")
            return

        print(f"\nFetched {len(markets)} markets")
        print("\nSample markets:")
        for i, market in enumerate(markets[:3]):
            print(f"{i+1}. {market.get('question', 'Unknown')}")

        print("\n" + "=" * 80)
        print("Step 2: Extracting asset IDs")
        print("=" * 80)

        first_market = markets[0]
        print(f"\nSelected market: {first_market.get('question', 'Unknown')}")

        clob_token_ids = json.loads(first_market["clobTokenIds"])
        print(f"Asset IDs: {clob_token_ids}")

        print("\n" + "=" * 80)
        if args.mode == MODE_WEBSOCKET:
            print("Step 3: Starting WebSocket client (Mode 0)")
        else:
            print("Step 3: Starting REST API poller (Mode 1)")
        print("=" * 80)

        if args.mode == MODE_WEBSOCKET:
            # WebSocket mode - stream price changes
            client = PolymarketWebSocketClient(asset_ids=clob_token_ids, channel=MARKET_CHANNEL)
            client.run()
        else:
            # REST API mode - poll order book depth
            poller = OrderBookPoller(asset_ids=clob_token_ids, poll_interval=args.poll_interval)
            poller.poll_order_books()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
