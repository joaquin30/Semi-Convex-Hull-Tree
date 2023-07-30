CXX = g++
FLAGS = -std=c++11 -Wall -Wextra -Wshadow -pedantic-errors -I.
TEST_FLAGS = -Og -ggdb3 -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC

all: test.exe

test: test.exe
	./test.exe test/dataset.csv

test.exe: test/test.cpp schtree.h
	$(CXX) -o test.exe test/test.cpp $(FLAGS) $(TEST_FLAGS)

clean:
	rm test.exe

.PHONY: all test clean
