#include <stdio.h>
#include <stdlib.h>

int main() {
	int stride = 2; // filter stride length
	int nx = 8, ny = 8, nz = 3; // input dimensions
	int f = 2; // number of filters
	int sx = 4, sy = 4; // filter dimensions
	int ox = (nx-sx)/stride + 1; // output dimensions
	int oy = (ny-sy)/stride + 1;

	int ***neuron = (int***)malloc(nz*sizeof(int**));
	for (int z = 0; z < nz; z++) {
		neuron[z] = (int**)malloc(ny*sizeof(int*));
		for (int y = 0; y < ny; y++) {
			neuron[z][y] = (int*)malloc(nx*sizeof(int));
		}
	}

	int ****synapse = (int****)malloc(f*sizeof(int***));
	for (int i = 0; i < f; i++) {
		synapse[i] = (int***)malloc(nz*sizeof(int**));
		for (int z = 0; z < nz; z++) {
			synapse[i][z] = (int**)malloc(sy*sizeof(int*));
			for (int y = 0; y < sy; y++) {
				synapse[i][z][y] = (int*)malloc(sx*sizeof(int));
			}
		}
	}

	int ***output = (int***)malloc(f*sizeof(int**));
	for (int i = 0; i < f; i++) {
		output[i] = (int**)malloc(oy*sizeof(int*));
		for (int y = 0; y < oy; y++) {
			output[i][y] = (int*)malloc(ox*sizeof(int));
		}
	}

	FILE *fp;
	fp = fopen("input", "r");

	// collect neuron data
	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				fscanf(fp, "%d", &neuron[k][j][i]);
			}
		}
	}
	// collect synapse data
	for (int l = 0; l < f; l++) {
		for (int k = 0; k < nz; k++) {
			for (int j = 0; j < sy; j++) {
				for (int i = 0; i < sx; i++) {
					fscanf(fp, "%d", &synapse[l][k][j][i]);
				}
			}
		}
	}

	fclose(fp);

	// perform convolutional layer computation
	for (int oi = 0; oi < ox; oi++) {
		for (int oj = 0; oj < oy; oj++) {
			for (int ok = 0; ok < f; ok++) {
				int sum = 0;
				int xstart = oi*stride;
				int ystart = oj*stride;
				for (int i = 0; i < sx; i++) {
					for (int j = 0; j < sy; j++) {
						for (int k = 0; k < nz; k++) {
							sum += synapse[ok][k][j][i]*neuron[k][j+ystart][i+xstart];
						}
					}
				}
				output[ok][oj][oi] = sum;
			}
		}
	}

	// print output
	for (int k = 0; k < f; k++) {
		for (int j = 0; j < ox; j++) {
			for (int i = 0; i < oy; i++) {
				printf("%d ", output[k][j][i]);
			}
			printf("\n");
		}
		printf("\n");
	}

	return 0;
}
