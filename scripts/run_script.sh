#!/bin/bash
set -e

make

./build/main echo.websocket.org 443 "hi"
