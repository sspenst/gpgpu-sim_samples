#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <time.h>

#define SIZE(A) A*sizeof(int)
#define FSIZE(A) A*sizeof(float)
#define LENGTH 64 // max threads is 2048
#define CTA LENGTH 
#define VERBOSE 0

// SST the matrix
__global__ void SSTMatrix(float* M, int* maddr, int N) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	if (tid < N) {
		for (int i = bid; i < N; i += CTA) {
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
__global__ void SMSV(float* M, float* V, float* R, int* maddr, int* addr, int N) {
	int tid = threadIdx.x;
	if (tid < N) {
		__shared__ int psum[LENGTH];
		psum[tid] = 0; // initialize psum with 0s
		__syncthreads(); // psum is finished being written to
		int numCols = (int)(*addr - (intptr_t)&V[0])/4; // end of SST for vector
		for (int i = 0; i <= numCols; i++) { // loop through columns
			int vid = (int)V[i+N]; // vector index
			int cEnd = (int)(maddr[vid] - (intptr_t)&M[2*N*vid])/4; // end of SST for column
			if (tid <= cEnd) {
				int mid = (int)M[2*N*vid + tid+N]; // matrix index
				psum[mid] += M[2*N*vid + tid] * V[i];
			}
			__syncthreads();
		}
		R[tid] = psum[tid];
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

	float *v_orig = (float*)calloc(LENGTH, FSIZE(1)); // keep track of the original vector
	float *m_orig = (float*)calloc(LENGTH*LENGTH, FSIZE(1)); // keep track of the original matrix
	float *h_matrix = (float*)calloc(2*LENGTH*LENGTH, FSIZE(1)); // x = LENGTH, y = 2*LENGTH
	float *h_vector = (float*)calloc(2*LENGTH, FSIZE(1)); // 2*LENGTH to store values as well as indices
	float *h_result = (float*)malloc(FSIZE(LENGTH));
	float *answer = (float*)calloc(LENGTH, FSIZE(1));
	unsigned m_zero = 0;
	unsigned v_zero = 0;

	// use this matrix in column major order
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			int r = rand() % 90 + 1; // 50% chance of zero
			if (r < 10) m_orig[i + j*LENGTH] = r;
			else m_zero++;
			h_matrix[i + j*2*LENGTH] = m_orig[i + j*LENGTH];
			h_matrix[i+LENGTH + j*2*LENGTH] = -1.0;
		}
	}
	for (int i = 0; i < LENGTH; i++) {
		int r = rand() % 90 + 1; // 70% chance of zero
		if (r < 10) v_orig[i] = r;
		else v_zero++;
		h_vector[i] = v_orig[i];
		h_vector[i+LENGTH] = -1.0; // initialize the second half of the array (indices) with -1
	}

	float *d_matrix, *d_vector, *d_result;
	int *d_maddr, *d_addr;
	cudaMalloc(&d_matrix, FSIZE(2*LENGTH*LENGTH));
	cudaMalloc(&d_vector, FSIZE(2*LENGTH));
	cudaMalloc(&d_result, FSIZE(LENGTH));
	cudaMalloc(&d_maddr, SIZE(LENGTH));
	cudaMalloc(&d_addr, SIZE(1));
	cudaMemcpy(d_matrix, h_matrix, FSIZE(2*LENGTH*LENGTH), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vector, h_vector, FSIZE(2*LENGTH), cudaMemcpyHostToDevice);

	SSTMatrix<<<CTA, LENGTH>>>(d_matrix, d_maddr, LENGTH);
	
	cudaMemcpy(h_matrix, d_matrix, FSIZE(2*LENGTH*LENGTH), cudaMemcpyDeviceToHost);
	
	
	SSTVector<<<1, LENGTH>>>(d_vector, d_addr, LENGTH);
	SMSV<<<1, LENGTH>>>(d_matrix, d_vector, d_result, d_maddr, d_addr, LENGTH);
	
	cudaMemcpy(h_matrix, d_matrix, FSIZE(2*LENGTH*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vector, d_vector, FSIZE(2*LENGTH), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, FSIZE(LENGTH), cudaMemcpyDeviceToHost);
	
	/*for (int i = 0; i < LENGTH; i++)
		for (int j = 0; h_matrix[j+LENGTH+i*2*LENGTH] != -1.0; j++)
			if (m_orig[(int)h_matrix[j+LENGTH+i*2*LENGTH] + i*LENGTH] != h_matrix[j+i*2*LENGTH])
				printf("\nSST MESSED UP\n");

	FILE *fp = fopen("data2.txt", "w+");
	fprintf(fp, "\n\nINTERMEDIATE MATRIX @$!$!#$!#$@!#^!<S-F6>^$@*(#$&(8\n\n");
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			fprintf(fp, "%d/%d\t", (int)h_matrix[j+i*2*LENGTH], (int)h_matrix[j+LENGTH+i*2*LENGTH]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n\nINTERMEDIATE VECTOR\n\n");
	for (int i = 0; i < LENGTH; i++) {
		fprintf(fp, "%d/%d ", (int)h_vector[i], (int)h_vector[i+LENGTH]);
	}

	fprintf(fp, "\n\nPSUSMSSDMFIP\n");
	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			fprintf(fp, "%d:%d ", j, (int)h_psum[i*LENGTH + j]);
		}
		fprintf(fp, "\n");
	}

	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			if ((int)m_orig[i+j*LENGTH] != 0 && (int)v_orig[j] != 0) {
				fprintf(fp, "%d:%d ", j, (int)m_orig[i+j*LENGTH]*(int)v_orig[j]);
			}
		}
		fprintf(fp, "\n");
	}

	for (int i = 0; i < LENGTH; i++) {
		for (int j = 0; j < LENGTH; j++) {
			if ((int)m_orig[i+j*LENGTH] != 0 && (int)v_orig[j] != 0) {
				bool exists = false;
				for (int k = 0; k < LENGTH; k++) {
					if ((int)h_psum[i*LENGTH+k] == (int)m_orig[i+j*LENGTH]*(int)v_orig[j]) exists = true;
				}
				if (exists == false) fprintf(fp, "PSUM MESSED UP: %d %d %d %d\n", i, j, (int)m_orig[i+j*LENGTH], (int)v_orig[j]);
			}
		}
	}
	fclose(fp);*/

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
	
	printf("\nCorrect result:\n");
#endif
	matMul(m_orig, v_orig, answer, LENGTH, LENGTH);
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

	printf("\nMatrix zero percentage: %f%%\n", (float)m_zero/(float)LENGTH/(float)LENGTH*100);
	printf("Vector zero percentage: %f%%\n", (float)v_zero/(float)LENGTH*100);

	if (correct == false) printf("\nWARNING: Incorrect result...\n");
	else printf("\nCorrect result!\n");
	
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
	free(answer);

	return 0;
}
