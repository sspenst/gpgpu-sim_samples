#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define THREADS 64
#define WARP_SIZE 32
#define WARPS ((THREADS+WARP_SIZE-1)/WARP_SIZE)
#define IP 4
#define OP 4
#define INBUFFERS 4
#define OUTBUFFERS (((WARP_SIZE/IP)*INBUFFERS+(WARP_SIZE/OP)-1)/(WARP_SIZE/OP))

__global__ void BSMAD(int *buffer, int *synapse, int *output) {
	volatile int tid = threadIdx.x;
	volatile int result = 0;
	asm("/*");
	asm("CPTX_BEGIN");
	asm("bsmad.s32 %0, %1, %2, %3, %4, %5, %6, %7, %8;" : "=r"(result) : "r"(IP), "r"(OP), "r"(buffer[tid*INBUFFERS]), "r"(buffer[tid*INBUFFERS+1]), "r"(buffer[tid*INBUFFERS+2]), "r"(buffer[tid*INBUFFERS+3]), "r"(synapse[tid]), "r"(output[(tid%WARP_SIZE)%OUTBUFFERS + OUTBUFFERS*(tid/WARP_SIZE)]));
	asm("CPTX_END");
	asm("*/");
	if (tid%WARP_SIZE < OUTBUFFERS) output[tid%WARP_SIZE + OUTBUFFERS*(tid/WARP_SIZE)] = result;
}

int main(int argc, char** argv) {
	int *h_buffer = (int*)malloc(INBUFFERS*THREADS*sizeof(int));
	int *h_synapse = (int*)malloc(THREADS*sizeof(int));
	int *h_output = (int*)calloc(WARPS*OUTBUFFERS,sizeof(int));

	for (int i = 0; i < INBUFFERS*THREADS; i++) { h_buffer[i] = 0xFDB97531; }
	for (int i = 0; i < THREADS; i++) { h_synapse[i] = 1 + 2*(i/WARP_SIZE); }
	for (int i = 0; i < WARPS*OUTBUFFERS; i++) { h_output[i] = 0x87654321; }

	int *d_buffer, *d_synapse, *d_output;
	cudaMalloc(&d_buffer, INBUFFERS*THREADS*sizeof(int));
	cudaMalloc(&d_synapse, THREADS*sizeof(int));
	cudaMalloc(&d_output, WARPS*OUTBUFFERS*sizeof(int));
	cudaMemcpy(d_buffer, h_buffer, INBUFFERS*THREADS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_synapse, h_synapse, THREADS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, WARPS*OUTBUFFERS*sizeof(int), cudaMemcpyHostToDevice);
	
	BSMAD<<<1, THREADS>>>(d_buffer, d_synapse, d_output);
	
	cudaMemcpy(h_output, d_output, WARPS*OUTBUFFERS*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < WARPS*OUTBUFFERS; i++) {
		if (i % OUTBUFFERS == 0) printf("\nWarp %d:\n", i/OUTBUFFERS);
		printf("%d: %08x\n", i, h_output[i]);
	}

	cudaFree(d_buffer); cudaFree(d_synapse); cudaFree(d_output);
	free(h_buffer); free(h_synapse); free(h_output);

	return 0;
}
