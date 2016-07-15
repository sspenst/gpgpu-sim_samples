SM = 20

all: dmdv

dmdv: dmdv.cu
	nvcc -I ~/samples/7.5/common/inc -ccbin g++ -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM) -lcudart -o dmdv dmdv.cu

clean: 
	rm -f dmdv _cuobjdump* _ptx* gpgpu_inst_stats.txt gpgpusim_power_report__*
