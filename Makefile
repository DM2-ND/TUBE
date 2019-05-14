CPP = g++

# Standard
FLAGS = -std=gnu++11
# Optimization
FLAGS += -Ofast -flto -march=native -funroll-loops
# # Warning
FLAGS += -Wall -Wno-unused-result
# # Msic
FLAGS += -lm -pthread
# FLAGS += -g -O0


all: tube

tube : tube.cpp
	$(CPP) tube.cpp -o tube $(FLAGS)

clean: 
	rm tube
