#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>

#define SIZE(A) A*sizeof(int)
#define MULTIPLIER_SIZE 32
#define NUM_MULTIPLIERS 32 // arbitrary size

typedef struct {
	int bits; // bit precision of neurons
	int stride; // filter stride length
	int nx, ny, nz; // input dimensions
	int f; // number of filters
	int sx, sy; // filter dimensions
	int ox, oy; // output dimensions
} conv_data;

__global__ void CLCTest(int *neuron, int *synapse, int *output, conv_data cd) {
	int tid = threadIdx.x;
	
	int *window = neuron + (tid/cd.ox)*cd.nx*cd.stride + (tid%cd.ox)*cd.stride; // addr = base + row offset + col offset (multiply by 4?)
	int out_neuron;
	int window_dim4 = cd.sx + (cd.sy << 8) + (cd.nz << 16) + (cd.ny << 24);

	for(int i = 0; i < cd.sx*cd.sy*cd.nz; i += NUM_MULTIPLIERS) {
		asm("/*");
		asm("CPTX_BEGIN");
		asm("clc.sstarr.s32 %0, [%1], [%2], %3, %4;" : "=r"(out_neuron) : "l"(window), "l"(synapse), "r"(window_dim4), "r"(cd.bits)); // perform CLC instruction
		/* 
		 * How to implement in GPGPU-Sim:
		 * Not necessary to worry about doing the computations 1 bit at a time (except maybe if you want to add the timing model).
		 * I'm using cd.* instead of bit shifting window_dim4 because it's nicer.
		 * for (int i = 0, j = 0; i < NUM_MULTIPLIERS; i++, j++) {
		 * 	if (cd.sy * cd.nz % i == 0) // if you have finished a slice of neurons
		 * 		j += (cd.ny - cd.sy) * cd.nz; // get the address of the next slice
		 * 	out_neuron += window[j] * synapse[j];
		 * }
		 * Need to also use cd.sx to check if we are still within the window.
		 * We should break from the for loop at that point.
		 * At this point you should have your answer.
		 */
		asm("CPTX_END");
		asm("*/");

		synapse += NUM_MULTIPLIERS; // multiply by 4?
		output[tid] = out_neuron;
	}
}

int main(int argc, char** argv) {
	conv_data cd;
	cd.bits = 4; cd.stride = 2;
	cd.nx = 8; cd.ny = 8; cd.nz = 3; cd.f = 2;
	cd.sx = 4; cd.sy = 4; cd.ox = 3; cd.oy = 3;

	int neuron[cd.nx][cd.ny][cd.nz];
	int synapse[cd.f][cd.sx][cd.sy][cd.nz];
	int output[cd.ox][cd.oy][cd.f];
	memset(output, 0, SIZE(cd.ox*cd.oy*cd.f));

	FILE *fp;
	fp = fopen("input", "r");

	// collect neuron data
	for (int k = 0; k < cd.nz; k++) {
		for (int j = 0; j < cd.ny; j++) {
			for (int i = 0; i < cd.nx; i++) {
				fscanf(fp, "%d", &neuron[i][j][k]);
			}
		}
	}
	// collect synapse data
	for (int l = 0; l < cd.f; l++) {
		for (int k = 0; k < cd.nz; k++) {
			for (int j = 0; j < cd.sy; j++) {
				for (int i = 0; i < cd.sx; i++) {
					fscanf(fp, "%d", &synapse[l][i][j][k]);
				}
			}
		}
	}
	
	int *d_neuron, *d_synapse, *d_output;
	cudaMalloc(&d_neuron, SIZE(cd.nx*cd.ny*cd.nz));
	cudaMalloc(&d_synapse, SIZE(cd.f*cd.sx*cd.sy*cd.nz));
	cudaMalloc(&d_output, SIZE(cd.ox*cd.oy*cd.f));
	cudaMemcpy(d_neuron, neuron, SIZE(cd.nx*cd.ny*cd.nz), cudaMemcpyHostToDevice);
	cudaMemcpy(d_synapse, synapse, SIZE(cd.f*cd.sx*cd.sy*cd.nz), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, SIZE(cd.ox*cd.oy*cd.f), cudaMemcpyHostToDevice);

	CLCTest<<<1, cd.ox*cd.oy>>>(d_neuron, d_synapse, d_output, cd);
	
	cudaMemcpy(output, d_output, SIZE(cd.ox*cd.oy*cd.f), cudaMemcpyDeviceToHost);
	
	// print output
	for (int k = 0; k < cd.f; k++) {
		for (int j = 0; j < cd.ox; j++) {
			for (int i = 0; i < cd.oy; i++) {
				printf("%d ", output[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}

	cudaFree(d_neuron);
	cudaFree(d_synapse);
	cudaFree(d_output);

	return 0;
}
