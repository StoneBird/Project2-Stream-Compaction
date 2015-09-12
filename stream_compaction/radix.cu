#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include <stream_compaction/thrust.h>

namespace RadixSort {

	__global__ void getTotalFalse(int *oB, int *oE, const int *idata, const int currentPass, const int n){
		int k = (blockIdx.x*blockDim.x + threadIdx.x) % n;

		int digit = 0;
		int dec = idata[k];
		for (int i = 0; i <= currentPass; i++){
			digit = dec % 2;
			dec = dec / 2;
		}

		if (digit == 0){
			oE[k] = 1;
			oB[k] = 0;
		} else {
			oB[k] = 1;
			oE[k] = 0;
		}
	}

	__global__ void getT(int *oT, const int *iF, const int totalFalses, const int n){
		int k = (blockIdx.x*blockDim.x + threadIdx.x) % n;
		oT[k] = k - iF[k] + totalFalses;
	}

	__global__ void rearrange(int *odata, int *idata, int *oB, int *oT, int *oF, const int n){
		int k = (blockIdx.x*blockDim.x + threadIdx.x) % n;
		int d = oB[k] == 1 ? oT[k] : oF[k];
		odata[d] = idata[k];
	}

int sort(int n, int *odata, const int *idata, const int passes) {
	int blockSize = 64;
	int gridSize = ceil(n / blockSize);

	int *dv_in, *dv_out;
	cudaMalloc((void**)&dv_in, n*sizeof(int));
	cudaMalloc((void**)&dv_out, n*sizeof(int));
	cudaMemcpy(dv_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	int *dv_b, *dv_e, *dv_t, *dv_f;
	cudaMalloc((void**)&dv_b, n*sizeof(int));
	cudaMalloc((void**)&dv_e, n*sizeof(int));
	cudaMalloc((void**)&dv_t, n*sizeof(int));
	cudaMalloc((void**)&dv_f, n*sizeof(int));

	int *hs_e, *hs_f;
	hs_e = (int *)malloc(n*sizeof(int));
	hs_f = (int *)malloc(n*sizeof(int));
	for (int p = 0; p <passes; p++){
		getTotalFalse<<<1, n >>>(dv_b, dv_e, dv_in, p, n);

		cudaMemcpy(hs_e, dv_e, n*sizeof(int), cudaMemcpyDeviceToHost);
		StreamCompaction::Thrust::scan(n, hs_f, hs_e);
		cudaMemcpy(dv_f, hs_f, n*sizeof(int), cudaMemcpyHostToDevice);

		int totalFalse = hs_e[n - 1] + hs_f[n - 1];

		getT<<<1, n >>>(dv_t, dv_f, totalFalse, n);

		rearrange << <1, n >> >(dv_out, dv_in, dv_b, dv_t, dv_f, n);
		cudaMemcpy(dv_in, dv_out, n*sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(odata, dv_out, n*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dv_in);
	cudaFree(dv_out);
	cudaFree(dv_b);
	cudaFree(dv_e);
	cudaFree(dv_t);
	cudaFree(dv_f);
	free(hs_e);
	free(hs_f);

	return n;
}

}
