objects = main.o printinfo.o cl_rig.o

CXX = g++

all: $(objects)
	$(CXX) $(objects) /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp -o main

%.o: %.cpp
	$(CXX) -I/usr/local/apps/cuda/cuda-10.1/include/ -c $< -o $@ -std=c++11

clean:
	rm -f *.o main

tidy:
	clang-format -i *.cpp *.h *.cl