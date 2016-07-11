SM = 20

all: dmsv

dmsv: dmsv.cu
	nvcc -ccbin g++ -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM) -lcudart -o dmsv dmsv.cu

clean: 
	rm -f dmsv _cuobjdump* _ptx* gpgpu_inst_stats.txt
