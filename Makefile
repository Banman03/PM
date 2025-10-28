CXX := g++
CXXSTD := -std=c++20
WARN := -Wall
LDFLAGS := -lboost_system -lssl -lcrypto -lpthread

TARGET := main
SRC := gamma-connection/main.cpp
BUILD_DIR := build
OUT := $(BUILD_DIR)/$(TARGET)

RELEASE_FLAGS := -O3
DEBUG_FLAGS   := -g -O0

all: release

release: CXXFLAGS := $(CXXSTD) $(WARN) $(RELEASE_FLAGS)
release: $(OUT)

debug: CXXFLAGS := $(CXXSTD) $(WARN) $(DEBUG_FLAGS)
debug: $(OUT)

$(OUT): $(SRC) | $(BUILD_DIR)
	@echo "[*] Compiling: $(SRC) -> $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all release debug clean
