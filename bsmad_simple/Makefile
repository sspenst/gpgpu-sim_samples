SM = 20

all: bsmad_simple

bsmad_simple: bsmad_simple.cu
	nvcc -ccbin g++ -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM) -lcudart -o bsmad_simple bsmad_simple.cu

clean: 
	rm -f bsmad_simple _cuobjdump* _ptx* gpgpu_inst_stats.txt
