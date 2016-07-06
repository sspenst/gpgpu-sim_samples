#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define SIZE(A) A*sizeof(int)
#define FSIZE(A) A*sizeof(float)
#define LENGTH 8

// Test the new SST instruction's functionality
__global__ void SSTTest(float* V, float* R, int* addr, int N) {
	int i = threadIdx.x;
	if (i < N) {
		volatile int return_val = 0;
		float element = V[i];
		asm("/*");
		asm("CPTX_BEGIN");
		
		/*__shared__ float S[8];
		int ir = N-i-1;
		S[i] = V[i];
		__syncthreads();
		R[i] = S[ir];*/

		asm(".sstarr .align 4 .b8 _Z9MatrixMulPfS_Pii__sst_var[32];"); // initialize sst_array
		asm("mov.u64 %rd10, _Z9MatrixMulPfS_Pii__sst_var;"); // rd10 = sst_array
		asm("add.s64 %rd11, %rd10, %rd9;"); // sst_array[i]
		asm("sst.sstarr.f32 %r3, [%rd11], %r1, %f1;"); // store element in sst_array[i]
		__syncthreads();
		volatile int ir = N-i-1;
		asm("mul.wide.s32 %rd12, %r5, 4;"); // ir*4
		asm("add.s64 %rd13, %rd10, %rd12;"); // sst_array[ir]
		asm("ld.sstarr.f32 %f1, [%rd13];"); // load from sst_array[ir] (replace element)
		R[i] = element;
		
		asm("CPTX_END");
		asm("*/");
		if (return_val != 0) *addr = return_val;
	}
}

int main(int argc, char** argv) {
	float *h_vector = (float*)calloc(LENGTH, FSIZE(1));
	float *h_result = (float*)malloc(FSIZE(LENGTH));
	int *h_addr = (int*)malloc(SIZE(1));

	for (int i = 0; i < LENGTH; i++) {
		if (i % 2 == 1) h_vector[i] = 0.0;
		else h_vector[i] = i + 1.0;
	}

	float *d_vector, *d_result;
	cudaMalloc(&d_vector, FSIZE(LENGTH));
	cudaMalloc(&d_result, FSIZE(LENGTH));
	int *d_addr;
	cudaMalloc(&d_addr, SIZE(1));

	cudaMemcpy(d_vector, h_vector, FSIZE(LENGTH), cudaMemcpyHostToDevice);

	SSTTest<<<1, LENGTH>>>(d_vector, d_result, d_addr, LENGTH);
	
	cudaMemcpy(h_vector, d_vector, FSIZE(LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, FSIZE(LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_addr, d_addr, SIZE(1), cudaMemcpyDeviceToHost);

	for (int i = 0; i < LENGTH; i++) {
		printf("%f\t", h_vector[i]);
		printf("%f\n", h_result[i]);
	}

	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_addr);
	free(h_vector);
	free(h_result);
	free(h_addr);

	return 0;
}
