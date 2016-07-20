#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <time.h>

#define SIZE(A) A*sizeof(int)
#define FSIZE(A) A*sizeof(float)
#define LENGTH 512 // max threads is 2048
#define BLOCK_SIZE 32
#define VERBOSE 0

// SST the vector
__global__ void SSTVector(float* V, int* addr, int N) {
	int i = threadIdx.x;
	if (i < N) {
		int return_val = 0;
		float element = V[i];
		asm("/*");
		asm("CPTX_BEGIN");
		asm("sst.sstarr.f32 %0, [%1], %2, %3;" : "=r"(return_val) : "l"(&V[0]), "r"(i), "f"(element));
		asm("CPTX_END");
		asm("*/");
		if (return_val != 0) *addr = return_val;
	}
}

// perform the matrix-vector multiplication
__global__ void DMSV(float* M, float* V, float* R, int* addr, int N) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	if (tid < N) {
		__shared__ float Vs[2*LENGTH];
		for (int i = tid; i < tid+LENGTH; i+=BLOCK_SIZE) {
			Vs[i] = V[i];
			Vs[i+N] = V[i+N];
		}
		__syncthreads();
		
		int numCols = (int)(*addr - (intptr_t)&V[0])/4;
		float psum = 0.0;
		for (int i = 0; i <= numCols; i++) {
			int vid = (int)Vs[i+N];
			//int vid = (int)V[i+N];
			psum += M[N*vid + bid*BLOCK_SIZE + tid] * Vs[i];
			//psum += M[N*vid + bid*BLOCK_SIZE + tid] * V[i];
		}
		R[bid*BLOCK_SIZE + tid] = psum;
	}
}

// column major matrix-vector multiplication
void matMul(float* matrix, float* vector, float* output, int width, int height) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)	{
			output[i] += matrix[j * height + i] * vector[j];
		}
	}
}

int main(int argc, char** argv) {
	srand(time(NULL));

	float *original = (float*)calloc(LENGTH, FSIZE(1)); // keep track of the original data
	float *h_matrix = (float*)calloc(LENGTH*LENGTH, FSIZE(1));
	float *h_vector = (float*)calloc(2*LENGTH, FSIZE(1)); // 2*LENGTH to store values as well as indices
	float *h_result = (float*)malloc(FSIZE(LENGTH));
	int *h_addr = (int*)malloc(SIZE(1));
	float *answer = (float*)malloc(FSIZE(LENGTH));
	unsigned zero = 0;

	// use this matrix in column major order
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			int r = rand() % 18 + 1; // 50% chance of zero
			if (r < 10) h_matrix[i*LENGTH + j] = r;
		}
	}
	for (int i = 0; i < LENGTH; i++) {
		int r = rand() % 90 + 1; // 90% chance of zero
		//int r = rand() % 45 + 1; // 80% chance of zero
		//int r = rand() % 30 + 1; // 70% chance of zero
		if (r < 10) original[i] = r;
		else zero++;
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
	DMSV<<<LENGTH/BLOCK_SIZE, BLOCK_SIZE>>>(d_matrix, d_vector, d_result, d_addr, LENGTH);
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
	printf("\nCorrect result:\n");
#endif
	matMul(h_matrix, original, answer, LENGTH, LENGTH);
	bool correct = 1;
	for (int i = 0; i < LENGTH; i++) {
#if VERBOSE==1
		printf("%d", (int)answer[i]);
#endif
		if (answer[i] != h_result[i]) {
			correct = 0;
#if VERBOSE==1
			printf("\t%d\t%d\t%d", i, (int)answer[i], (int)h_result[i]);
#else
			break;
#endif
		}
#if VERBOSE==1
		printf("\n");
#endif
	}

	printf("\nVector zero percentage: %f%%\n", (float)zero/(float)LENGTH*100);

	if (correct == false) printf("\nWARNING: Incorrect result...\n");
	else printf("\nCorrect result!\n");
	
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
