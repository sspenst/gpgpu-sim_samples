SM = 20

all: smsv

smsv: smsv.cu
	nvcc -ccbin g++ -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM) -lcudart -o smsv smsv.cu

clean: 
	rm -f smsv _cuobjdump* _ptx* gpgpu_inst_stats.txt
