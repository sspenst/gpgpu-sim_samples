#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <time.h>

#define SIZE(A) A*sizeof(int)
#define FSIZE(A) A*sizeof(float)
#define LENGTH 8 // max length is 64
#define VERBOSE 0

// Test the new SST instruction's functionality
__global__ void SSTTest(float* V, int* addr, int N) {
	int i = threadIdx.x;
	if (i < N) {
		int return_val = 0;
		float element = V[i];
		asm("/*");
		asm("CPTX_BEGIN");
		asm("sst.sstarr.f32 %0, [%1], %2, %3;" : "=r"(return_val) : "l"(&V[0]), "r"(i), "f"(element)); // perform SST instruction
		asm("CPTX_END");
		asm("*/");
		if (return_val != 0) *addr = (int)(return_val - (intptr_t)&V[0])/4; // last thread stores the result
	}
}

int main(int argc, char** argv) {
	float *original = (float*)calloc(LENGTH, FSIZE(1)); // keep track of the original data
	float *h_vector = (float*)calloc(2*LENGTH, FSIZE(1)); // 2*LENGTH to store values as well as indices
	int *h_addr = (int*)malloc(SIZE(1));

	for (int i = 0; i < LENGTH; i++) {
		if (i % 2 == 1) original[i] = 0.0;
		else original[i] = i + 1.0;
	}
	for (int i = 0; i < LENGTH; i++) {
		h_vector[i] = original[i];
		h_vector[i+LENGTH] = -1.0; // initialize the second half of the array (indices) with -1
	}

	float *d_vector;
	int *d_addr;
	cudaMalloc(&d_vector, FSIZE(2*LENGTH));
	cudaMalloc(&d_addr, SIZE(1));
	cudaMemcpy(d_vector, h_vector, FSIZE(2*LENGTH), cudaMemcpyHostToDevice);
	SSTTest<<<1, LENGTH>>>(d_vector, d_addr, LENGTH);
	cudaMemcpy(h_vector, d_vector, FSIZE(2*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_addr, d_addr, SIZE(1), cudaMemcpyDeviceToHost);

#if VERBOSE==1
	// output results
	printf("\nOriginal:\tResult: \tIndices:\n");
	for (int i = 0; i < LENGTH; i++) {
		printf("%f\t%f\t%d\n", original[i], h_vector[i], (int)h_vector[i+LENGTH]);
	}
	printf("\nSST return value: %d\n", h_addr[0]);
	printf("Array start: %p\nArray end: %p\n", &h_vector[0], &h_vector[h_addr[0]]);
#endif

	cudaFree(d_vector);
	cudaFree(d_addr);
	free(original);
	free(h_vector);
	free(h_addr);

	return 0;
}
