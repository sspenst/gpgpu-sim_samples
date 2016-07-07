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
		// these two instructions seem to be necessary to initialize the sstarr memory
		// is it possible to do these within sst_impl?
		asm(".sstarr .align 4 .b8 _Z9MatrixMulPfS_Pii__sst_var[32];"); // initialize sst_array
		asm("mov.u64 %rd10, _Z9MatrixMulPfS_Pii__sst_var;"); // rd10 = sst_array
		// perform SST instruction
		asm("sst.sstarr.f32 %r3, [%rd6], %r1, %f1;");
		asm("st.local.u32 [%rd1], %r3;");
		asm("CPTX_END");
		asm("*/");
		__syncthreads();
		R[i] = element;
		if (return_val != 0) *addr = return_val;
	}
}

int main(int argc, char** argv) {
	float *h_vector = (float*)calloc(2*LENGTH, FSIZE(1)); // twice as long to store values as well as indices
	float *h_result = (float*)malloc(FSIZE(LENGTH));
	int *h_addr = (int*)malloc(SIZE(1));

	for (int i = 0; i < LENGTH; i++) {
		if (i % 2 == 1) h_vector[i] = 0.0;
		else h_vector[i] = i + 1.0;
		h_vector[i+LENGTH] = 0.0;
	}

	float *d_vector, *d_result;
	cudaMalloc(&d_vector, FSIZE(2*LENGTH));
	cudaMalloc(&d_result, FSIZE(LENGTH));
	int *d_addr;
	cudaMalloc(&d_addr, SIZE(1));

	cudaMemcpy(d_vector, h_vector, FSIZE(2*LENGTH), cudaMemcpyHostToDevice);

	SSTTest<<<1, LENGTH>>>(d_vector, d_result, d_addr, LENGTH);
	
	cudaMemcpy(h_vector, d_vector, FSIZE(2*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, FSIZE(LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_addr, d_addr, SIZE(1), cudaMemcpyDeviceToHost);

	printf("\nValues: \tIndices:\n");
	for (int i = 0; i < LENGTH; i++) {
		printf("%f\t%d\n", h_vector[i], (int)h_vector[i+LENGTH]);
	}
	printf("\nSST return value: %d\n", h_addr[0]);
	printf("Array start: %p\nArray end: %p\n", &h_vector[0], &h_vector[h_addr[0]]);

	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_addr);
	free(h_vector);
	free(h_result);
	free(h_addr);

	return 0;
}
