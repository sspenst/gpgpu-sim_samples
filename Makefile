SM = 20

all: sst_simple

sst_simple: sst_simple.cu
	nvcc -ccbin g++ -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM) -lcudart -o sst_simple sst_simple.cu

clean: 
	rm -f sst_simple _cuobjdump* _ptx*
