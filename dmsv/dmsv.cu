#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <time.h>

#define SIZE(A) A*sizeof(int)
#define FSIZE(A) A*sizeof(float)
#define LENGTH 64 // max length is 64
#define VERBOSE 1

// SST the vector
__global__ void SSTVector(float* V, int* addr, int N) {
	int i = threadIdx.x;
	if (i < N) {
		int return_val = 0;
		float element = V[i];
		asm("/*");
		asm("CPTX_BEGIN");
		asm("sst.sstarr.f32 %0, [%1], %2, %3;" : "=r"(return_val) : "l"(&V[0]), "r"(i), "f"(element)); // perform SST instruction
		asm("CPTX_END");
		asm("*/");
		if (return_val != 0) *addr = return_val;
		//if (return_val != 0) *addr = (int)(return_val - (intptr_t)&V[0])/4; // last thread stores the result
	}
}

// perform the matrix-vector multiplication
__global__ void DMSV(float* M, float* V, float* R, int* addr, int N) {
	int tid = threadIdx.x;
	if (tid < N) {
		int numCols = *addr - (intptr_t)&V[0] + 1;
		float psum = 0.0;
		for (int i = 0; i < numCols; i++) {
			int vid = (int)V[i+N];
			psum += M[N*vid + tid] * V[i];
		}
		R[tid] = psum;
	}
}

int main(int argc, char** argv) {
	srand(time(NULL));

	float *original = (float*)calloc(LENGTH, FSIZE(1)); // keep track of the original data
	float *h_matrix = (float*)calloc(LENGTH*LENGTH, FSIZE(1));
	float *h_vector = (float*)calloc(2*LENGTH, FSIZE(1)); // 2*LENGTH to store values as well as indices
	float *h_result = (float*)malloc(FSIZE(LENGTH));
	int *h_addr = (int*)malloc(SIZE(1));

	// use this matrix in column major order
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			int r = rand() % 18 + 1; // 50% chance of zero
			if (r < 10) h_matrix[i*LENGTH + j] = r;
		}
	}
	for (int i = 0; i < LENGTH; i++) {
		int r = rand() % 30 + 1; // 70% chance of zero
		if (r < 10) original[i] = r;
	}
	for (int i = 0; i < LENGTH; i++) {
		h_vector[i] = original[i];
		h_vector[i+LENGTH] = -1.0; // initialize the second half of the array (indices) with -1
	}

	float *d_matrix, *d_vector, *d_result;
	int *d_addr;
	cudaMalloc(&d_matrix, FSIZE(LENGTH*LENGTH));
	cudaMalloc(&d_vector, FSIZE(2*LENGTH));
	cudaMalloc(&d_result, FSIZE(LENGTH));
	cudaMalloc(&d_addr, SIZE(1));
	cudaMemcpy(d_matrix, h_matrix, FSIZE(LENGTH*LENGTH), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vector, h_vector, FSIZE(2*LENGTH), cudaMemcpyHostToDevice);
	SSTVector<<<1, LENGTH>>>(d_vector, d_addr, LENGTH);
	DMSV<<<1, LENGTH>>>(d_matrix, d_vector, d_result, d_addr, LENGTH);
	cudaMemcpy(h_matrix, d_matrix, FSIZE(LENGTH*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vector, d_vector, FSIZE(2*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, FSIZE(LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_addr, d_addr, SIZE(1), cudaMemcpyDeviceToHost);

#if VERBOSE==1
	// output results
	printf("\nMatrix * Vector:\n");
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			printf("%d ", (int)h_matrix[i + j*LENGTH]);
		}
		printf("\t*\t%d\n", (int)original[i]);
	}

	printf("\nResult:\n");
	for (int i = 0; i < LENGTH; i++) {
		printf("%d\n", (int)h_result[i]);
	}
#endif
	
	cudaFree(d_matrix);
	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_addr);
	free(original);
	free(h_matrix);
	free(h_vector);
	free(h_result);
	free(h_addr);

	return 0;
}
