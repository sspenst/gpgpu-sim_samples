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
		//asm(".shared .align 4 .b8 _Z9MatrixMulPfS_Pii__sst_var[4];");
		//asm("sst.shared.f32 %r3, [_Z9MatrixMulPfS_Pii__sst_var], %r1, %f1;");
		
		asm(".sstarr .align 4 .b8 _Z9MatrixMulPfS_Pii__sst_var[32];");
		asm("mov.u64 %rd11, _Z9MatrixMulPfS_Pii__sst_var;");
		asm("add.s64 %rd12, %rd11, %rd7;");
		asm("sst.sstarr.f32 %r3, [%rd12], %r1, %f1;");
		//asm("sst.sstarr.f32 %r3, [_Z9MatrixMulPfS_Pii__sst_var], %r1, %f1;");

		//asm("sst.sstarr.f32 %0, [_Z9MatrixMulPfS_Pii__sst_var], %1, %2;" : "=r"(return_val) : "r"(i) : "r"(element));
		// for now, manually figure out what these registers are to be able to test sst function:
		// %r3 = register of return_val, need to also run st.local.u32 [%rd1], %r3;
		// %rd6 = register of V
		// %r1 = register of i
		// %f1 = register of element
		asm("CPTX_END");
		asm("*/");
		R[i] = element;
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
