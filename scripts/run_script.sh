#!/bin/bash
set -e

make

./build/main ws-subscriptions-clob.polymarket.com 443 "hi"
