#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>

#define SIZE(A) A*sizeof(int)

// for this method to work, (MULTIPLIER_SIZE <= NUM_MULTIPLIERS) must be true
#define MULTIPLIER_SIZE 32
#define NUM_MULTIPLIERS 32 // arbitrary size

typedef struct {
	int ibits, obits; // bit precision of neurons
	int stride; // filter stride length
	int nx, ny, nz; // input dimensions
	int f; // number of filters
	int sx, sy; // filter dimensions
	int ox, oy; // output dimensions
} conv_data;

__global__ void CLCTest(int *neuron, int *synapse, int *output, conv_data cd) {
	volatile int tid = threadIdx.x;
	int bid = blockIdx.x;
	bool last_block = (cd.ox*cd.oy / NUM_MULTIPLIERS + ((cd.ox*cd.oy) % NUM_MULTIPLIERS != 0) - 1) == bid;

	int window_size = cd.sx*cd.sy*cd.nz;
	int num_windows = cd.ox*cd.oy;

	for (int i = 0; i < window_size/NUM_MULTIPLIERS; i++) {
		volatile int out_neuron;
		volatile int j;

		// send groups of 4 neurons to the operand collector
		for (j = bid*MULTIPLIER_SIZE+3; (j < (bid+1)*MULTIPLIER_SIZE) && (j < num_windows); j += 4) {
			int temp[4];
			for (int k = 0; k < 4; k++) {
				temp[k] = i*NUM_MULTIPLIERS+tid + ((j-3+k)/cd.oy)*cd.ny*cd.nz*cd.stride + ((j-3+k)%cd.oy)*cd.nz*cd.stride;
			}

			asm("/*");
			asm("CPTX_BEGIN");
			asm("clp.s32 %0, %1, %2, %3;" :: "r"(neuron[temp[0]]), "r"(neuron[temp[1]]), "r"(neuron[temp[2]]), "r"(neuron[temp[3]]));
			asm("CPTX_END");
			asm("*/");
		}

		// send the remaining neurons individually to the operand collector
		if (last_block) { // this assumes MULTIPLIER_SIZE % 4 == 0
			j -= 3; // revert the last iteration from previous for loop and add 1
			for (; j < num_windows; j++) {
				asm("/*");
				asm("CPTX_BEGIN");
				asm("clp.s32 %0;" :: "r"(neuron[i+tid + (j/cd.oy)*cd.ny*cd.nz*cd.stride + (j%cd.oy)*cd.nz*cd.stride]));
				asm("CPTX_END");
				asm("*/");
			}
		}

		// compute the result
		asm("/*");
		asm("CPTX_BEGIN");
		asm("clc.s32 %0, %1, %2, %3;" : "=r"(out_neuron) : "r"(synapse[i*NUM_MULTIPLIERS+tid]), "r"(cd.ibits), "r"(cd.obits));
		asm("CPTX_END");
		asm("*/");

		// store the result, making sure the garbage return values are ignored
		if ((!last_block && (tid < MULTIPLIER_SIZE)) || (last_block && (tid < num_windows % MULTIPLIER_SIZE))) {
			output[bid*MULTIPLIER_SIZE + tid] += out_neuron;
		}
	}
}

int main(int argc, char** argv) {
	conv_data cd;
	cd.ibits = 4; cd.obits = 8; cd.stride = 2;
	cd.nx = 8; cd.ny = 8; cd.nz = 3; cd.f = 2;
	cd.sx = 4; cd.sy = 4; cd.ox = 3; cd.oy = 3;
	// ox = (nx-sx)/stride + 1;
	// oy = (ny-sy)/stride + 1;

	int neuron[cd.nx*cd.ny*cd.nz];
	int synapse[cd.f*cd.sx*cd.sy*cd.nz];
	int output[cd.f*cd.ox*cd.oy];
	memset(output, 0, SIZE(cd.ox*cd.oy*cd.f));

	FILE *fp;
	fp = fopen("input", "r");

	// collect neuron data
	for (int k = 0; k < cd.nz; k++) {
		for (int j = 0; j < cd.ny; j++) {
			for (int i = 0; i < cd.nx; i++) {
				fscanf(fp, "%d", &neuron[i*cd.ny*cd.nz + j*cd.nz + k]);
			}
		}
	}
	// collect synapse data
	for (int l = 0; l < cd.f; l++) {
		for (int k = 0; k < cd.nz; k++) {
			for (int j = 0; j < cd.sy; j++) {
				for (int i = 0; i < cd.sx; i++) {
					fscanf(fp, "%d", &synapse[l*cd.sx*cd.sy*cd.nz + i*cd.sy*cd.nz + j*cd.nz + k]);
				}
			}
		}
	}

	fclose(fp);
	
	int *d_neuron, *d_synapse, *d_output;
	cudaMalloc(&d_neuron, SIZE(cd.nx*cd.ny*cd.nz));
	cudaMalloc(&d_synapse, SIZE(cd.f*cd.sx*cd.sy*cd.nz));
	cudaMalloc(&d_output, SIZE(cd.ox*cd.oy*cd.f));
	cudaMemcpy(d_neuron, neuron, SIZE(cd.nx*cd.ny*cd.nz), cudaMemcpyHostToDevice);
	cudaMemcpy(d_synapse, synapse, SIZE(cd.f*cd.sx*cd.sy*cd.nz), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, SIZE(cd.ox*cd.oy*cd.f), cudaMemcpyHostToDevice);

	// one thread block is composed of MULTIPLIER_SIZE windows and NUM_MULTIPLIERS threads
	int blocks = cd.ox*cd.oy / NUM_MULTIPLIERS + ((cd.ox*cd.oy) % NUM_MULTIPLIERS != 0);
	// apply one filter per kernel
	for (int f = 0; f < cd.f; f++) {
		CLCTest<<<blocks, NUM_MULTIPLIERS>>>(d_neuron, d_synapse + f*cd.sx*cd.sy*cd.nz, d_output + f*cd.ox*cd.oy, cd);
	}

	cudaMemcpy(output, d_output, SIZE(cd.ox*cd.oy*cd.f), cudaMemcpyDeviceToHost);

	// print output
	for (int k = 0; k < cd.f; k++) {
		for (int j = 0; j < cd.oy; j++) {
			for (int i = 0; i < cd.ox; i++) {
				printf("%d ", output[k*cd.ox*cd.oy + i*cd.oy + j]);
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
