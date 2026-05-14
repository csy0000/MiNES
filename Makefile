CXX      ?= clang++
CXXFLAGS ?= -O2 -std=c++17

SRC = simulations/cpp/neq_sim.cpp
BIN = simulations/cpp/neq_sim

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) $(wildcard src/cpp/*.h)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)
