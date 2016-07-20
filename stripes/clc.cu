#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>

#define SIZE(A) A*sizeof(int)
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
	int window_id = bid*MULTIPLIER_SIZE + tid;

	if (window_id < cd.ox*cd.oy) {
		int *window = neuron + (window_id/cd.ox)*cd.nx*cd.stride + (window_id%cd.ox)*cd.stride; // addr = base + row offset + col offset (multiply by 4?)
		int precision = cd.ibits + (cd.obits << 16);
		int window_size = cd.sx*cd.sy*cd.nz;
	
		for(int i = 0; i < window_size; i += NUM_MULTIPLIERS) {
			int out_neuron;
			asm("/*");
			asm("CPTX_BEGIN");
			// load synapses into operand collectors
			for (int j = i+tid; j < i+NUM_MULTIPLIERS && j < window_size; j += cd.ox*cd.oy) {
				asm("ldo.s32 %0, [%1];" : "=r"(j) : "r"(synapse[j]));
			}
			// load neurons into operand collectors
			for (int j = i; j < i+NUM_MULTIPLIERS && j < window_size; j++) {
				asm("ldo.s32 %0, [%1];" : "=r"(j) : "r"(window[j]));
			}
			asm("clc.s32 %0, %1, %2;" : "=r"(out_neuron) : "r"(tid), "r"(precision));
			asm("CPTX_END");
			asm("*/");
			output[tid] = out_neuron;
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

	int neuron[cd.nx][cd.ny][cd.nz]; // maybe change these to 1D
	int synapse[cd.f][cd.sx][cd.sy][cd.nz];
	int output[cd.f*cd.ox*cd.oy];
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

	fclose(fp);

	int *d_neuron, *d_synapse, *d_output;
	cudaMalloc(&d_neuron, SIZE(cd.nx*cd.ny*cd.nz));
	cudaMalloc(&d_synapse, SIZE(cd.f*cd.sx*cd.sy*cd.nz));
	cudaMalloc(&d_output, SIZE(cd.ox*cd.oy*cd.f));
	cudaMemcpy(d_neuron, neuron, SIZE(cd.nx*cd.ny*cd.nz), cudaMemcpyHostToDevice);
	cudaMemcpy(d_synapse, synapse, SIZE(cd.f*cd.sx*cd.sy*cd.nz), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, SIZE(cd.ox*cd.oy*cd.f), cudaMemcpyHostToDevice);

	// apply one filter per kernel
	for (int f = 0; f < cd.f; f++) {
		int blocks = cd.ox*cd.oy / MULTIPLIER_SIZE + ((cd.ox*cd.oy) % MULTIPLIER_SIZE != 0); // number of thread blocks required when there are MULTIPLIER_SIZE threads per block
		// this assumes there are NUM_MULTIPLIERS multipliers per thread block
		CLCTest<<<blocks, MULTIPLIER_SIZE>>>(d_neuron, d_synapse + f*cd.sx*cd.sy*cd.nz, d_output + f*cd.ox*cd.oy, cd);
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
