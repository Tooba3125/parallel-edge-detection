CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall
LDFLAGS = -fopenmp
TARGET = edgedetect

SOURCES = main.cpp edge_detection.cpp
OBJECTS = main.o edge_detection.o

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

main.o: main.cpp edge_detection.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

edge_detection.o: edge_detection.cpp edge_detection.h
	$(CXX) $(CXXFLAGS) -c edge_detection.cpp -o edge_detection.o

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f results/*.png

run: all
	./$(TARGET)

help:
	@echo "Commands:"
	@echo "  make       - Build the program"
	@echo "  make clean - Remove compiled files"
	@echo "  make run   - Build and run"
