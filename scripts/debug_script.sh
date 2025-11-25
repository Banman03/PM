#!/bin/bash
set -e

# Build the debug version
make debug

# Run gdb with program arguments properly set
gdb --args ./build/main 1764087109596 443 "hi"
