#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <time.h>

#define SIZE(A) A*sizeof(int)
#define FSIZE(A) A*sizeof(float)
#define LENGTH 8 // max length is 64
#define VERBOSE 1

// SST the matrix
__global__ void SSTMatrix(float* M, int* maddr, int N) {
	int tid = threadIdx.x;
	if (tid < N) {
		for (int i = 0; i < N; i++) {
			int return_val = 0;
			float element = M[i*2*N + tid];
			asm("/*");
			asm("CPTX_BEGIN");
			asm("sst.sstarr.f32 %0, [%1], %2, %3;" : "=r"(return_val) : "l"(&M[i*2*N]), "r"(tid), "f"(element));
			asm("CPTX_END");
			asm("*/");
			if (return_val != 0) maddr[i] = return_val;
		}
	}
}

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
__global__ void SMSV(float* M, float* V, float* R, float* P, int* maddr, int* addr, int N) {
	int tid = threadIdx.x;
	if (tid < N) {
		__shared__ int psumIndex[LENGTH];
		psumIndex[tid] = 0; // initialize psumIndex with 0s
		__syncthreads(); // psumIndex is finished being written to
		int numCols = (int)(*addr - (intptr_t)&V[0])/4; // end of SST for vector
		for (int i = 0; i <= numCols; i++) { // loop through columns
			int vid = (int)V[i+N]; // vector index
			int cEnd = (int)(maddr[vid] - (intptr_t)&M[2*N*vid])/4; // end of SST for column
			if (tid <= cEnd) {
				int mid = (int)M[2*N*vid + tid+N]; // matrix index (THIS IS -1 SOMEHOW???)
				P[psumIndex[mid] + N*mid] += M[2*N*vid + tid] * V[i]; // append M element * V element to correct P row
				psumIndex[mid]++;
			}
		}
		__syncthreads(); // P is finished being written to
		float psum = 0.0;
		for (int i = 0; i < psumIndex[tid]; i++) {
			psum += P[tid*N + i];
		}
		R[tid] = psum;
	}
}

int main(int argc, char** argv) {
	srand(time(NULL));

	float *v_orig = (float*)calloc(LENGTH, FSIZE(1)); // keep track of the original vector
	float *m_orig = (float*)calloc(LENGTH*LENGTH, FSIZE(1)); // keep track of the original matrix
	float *h_matrix = (float*)calloc(2*LENGTH*LENGTH, FSIZE(1)); // x = LENGTH, y = 2*LENGTH
	float *h_vector = (float*)calloc(2*LENGTH, FSIZE(1)); // 2*LENGTH to store values as well as indices
	float *h_result = (float*)malloc(FSIZE(LENGTH));
	float *h_psum = (float*)calloc(LENGTH*LENGTH, FSIZE(1)); // ensure psum matrix is initialized with 0s
	int *h_maddr = (int*)malloc(SIZE(LENGTH));

	// use this matrix in column major order
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			int r = rand() % 18 + 1; // 50% chance of zero
			if (r < 10) m_orig[i + j*LENGTH] = r;
			h_matrix[i + j*2*LENGTH] = m_orig[i + j*LENGTH];
			h_matrix[i+LENGTH + j*2*LENGTH] = -1.0;
		}
	}
	for (int i = 0; i < LENGTH; i++) {
		int r = rand() % 30 + 1; // 70% chance of zero
		if (r < 10) v_orig[i] = r;
		h_vector[i] = v_orig[i];
		h_vector[i+LENGTH] = -1.0; // initialize the second half of the array (indices) with -1
	}

	float *d_matrix, *d_vector, *d_result, *d_psum;
	int *d_maddr, *d_addr;
	cudaMalloc(&d_matrix, FSIZE(2*LENGTH*LENGTH));
	cudaMalloc(&d_vector, FSIZE(2*LENGTH));
	cudaMalloc(&d_result, FSIZE(LENGTH));
	cudaMalloc(&d_psum, FSIZE(LENGTH*LENGTH));
	cudaMalloc(&d_maddr, SIZE(LENGTH));
	cudaMalloc(&d_addr, SIZE(1));
	cudaMemcpy(d_matrix, h_matrix, FSIZE(2*LENGTH*LENGTH), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vector, h_vector, FSIZE(2*LENGTH), cudaMemcpyHostToDevice);
	cudaMemcpy(d_psum, h_psum, FSIZE(LENGTH*LENGTH), cudaMemcpyHostToDevice);

	SSTMatrix<<<1, LENGTH>>>(d_matrix, d_maddr, LENGTH);
	SSTVector<<<1, LENGTH>>>(d_vector, d_addr, LENGTH);
	SMSV<<<1, LENGTH>>>(d_matrix, d_vector, d_result, d_psum, d_maddr, d_addr, LENGTH);
	
	cudaMemcpy(h_matrix, d_matrix, FSIZE(2*LENGTH*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vector, d_vector, FSIZE(2*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, FSIZE(LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_maddr, d_maddr, SIZE(LENGTH), cudaMemcpyDeviceToHost);

#if VERBOSE==1
	// output results
	printf("\nMatrix * Vector:\n");
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			printf("%d ", (int)m_orig[i + j*LENGTH]);
		}
		printf("\t*\t%d\n", (int)v_orig[i]);
	}

	printf("\nResult:\n");
	for (int i = 0; i < LENGTH; i++) {
		printf("%d\n", (int)h_result[i]);
	}
#endif
	
	cudaFree(d_matrix);
	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_maddr);
	cudaFree(d_addr);
	free(v_orig);
	free(m_orig);
	free(h_matrix);
	free(h_vector);
	free(h_result);
	free(h_maddr);

	return 0;
}
