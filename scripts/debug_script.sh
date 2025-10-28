#!/bin/bash
set -e

# Build the debug version
make debug

# Run gdb with program arguments properly set
gdb --args ./build/main echo.websocket.org 443 "hi"
