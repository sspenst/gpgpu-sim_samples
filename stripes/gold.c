#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
	int bits = 4; // bit precision of neurons
	int stride = 2; // filter stride length
	int nx = 8, ny = 8, nz = 3; // input dimensions
	int f = 2; // number of filters
	int sx = 4, sy = 4; // filter dimensions
	int ox = 3, oy = 3; // output dimensions

	int neuron[nx][ny][nz];
	int synapse[f][sx][sy][nz];
	int output[ox][oy][f];
	memset(output, 0, ox*oy*f*sizeof(int));
	
	FILE *fp;
	fp = fopen("input", "r");

	// collect neuron data
	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				fscanf(fp, "%d", &neuron[i][j][k]);
			}
		}
	}
	// collect synapse data
	for (int l = 0; l < f; l++) {
		for (int k = 0; k < nz; k++) {
			for (int j = 0; j < sy; j++) {
				for (int i = 0; i < sx; i++) {
					fscanf(fp, "%d", &synapse[l][i][j][k]);
				}
			}
		}
	}

	// perform convolutional layer computation
	for (int oi = 0; oi < ox; oi++) {
		for (int oj = 0; oj < oy; oj++) {
			for (int ok = 0; ok < f; ok++) {
				// compute each output neuron
				for (int b = 0; b < bits; b++) {
					// perform the calculation one bit at a time
					int sum = 0;
					int xstart = oi*stride;
					int ystart = oj*stride;
					for (int i = 0; i < sx; i++) {
						for (int j = 0; j < sy; j++) {
							for (int k = 0; k < nz; k++) {
								sum += synapse[ok][i][j][k]*(neuron[i+xstart][j+ystart][k] & (1 << b));
							}
						}
					}
					output[oi][oj][ok] += sum;
				}
			}
		}
	}
	
	// print output
	for (int k = 0; k < f; k++) {
		for (int j = 0; j < ox; j++) {
			for (int i = 0; i < oy; i++) {
				printf("%d ", output[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}

	return 0;
}
