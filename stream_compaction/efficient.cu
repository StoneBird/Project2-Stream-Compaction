#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

	__global__ void scanUp(int d, int *idata){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k % (int)pow((double)2, (double)(d + 1)) == 0){
			idata[k - 1 + (int)pow((double)2, (double)(d + 1))] += idata[k - 1 + (int)pow((double)2, (double)d)];
		}
	}

	__global__ void scanDown(int d, int *idata){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k % (int)pow((double)2, (double)(d + 1)) == 0){
			int t = idata[k - 1 + (int)pow((double)2, (double)d)];
			idata[k - 1 + (int)pow((double)2, (double)d)] = idata[k - 1 + (int)pow((double)2, (double)(d + 1))];
			idata[k - 1 + (int)pow((double)2, (double)(d + 1))] += t;
		}
	}

	__global__ void filter(int *odata, int *idata){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (idata[k] == 0){
			odata[k] = 0;
		}
		else {
			odata[k] = 1;
		}
	}

	__global__ void scatter(int *odata, int *idata, int *filter, int *idx){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (filter[k] == 1){
			odata[idx[k]] = idata[k];
		}
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	// Padding
	int m = (int)pow((double)2, (double)ilog2ceil(n));
	int *pidata;
	pidata = (int*)malloc(m*sizeof(int));
	for (int i = 0; i < n; i++){
		pidata[i] = idata[i];
	}
	if (m > n){
		for (int i = n; i < m; i++){
			pidata[i] = 0;
		}
	}
	int *dev_pidata;
	cudaMalloc((void **)&dev_pidata, m*sizeof(int));
	cudaMemcpy(dev_pidata, pidata, m*sizeof(int), cudaMemcpyHostToDevice);
	// Scan
	for (int d = 0; d < ilog2ceil(m); d++){
		scanUp << <1, m>> >(d, dev_pidata);
	}
	cudaMemcpy(pidata, dev_pidata, m*sizeof(int), cudaMemcpyDeviceToHost);
	pidata[m - 1] = 0;
	cudaMemcpy(dev_pidata, pidata, m*sizeof(int), cudaMemcpyHostToDevice);
	for (int d = ilog2ceil(m)-1; d >=0; d--){
		scanDown<<<1, m>>>(d, dev_pidata);
	}
	cudaMemcpy(pidata, dev_pidata, m*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++){
		odata[i] = pidata[i];
	}
	cudaFree(dev_pidata);
	free(pidata);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
	int *f;
	int *dev_idata;
	cudaMalloc((void**)&f, n * sizeof(int));
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	filter << <1, n >> >(f, dev_idata);

	int *hs_idx;
	int *hs_f;
	hs_idx = (int *)malloc(n * sizeof(int));
	hs_f = (int *)malloc(n*sizeof(int));
	cudaMemcpy(hs_f, f, n * sizeof(int), cudaMemcpyDeviceToHost);
	scan(n, hs_idx, hs_f);

	int *idx;
	int *dv_out;
	cudaMalloc((void**)&idx, n * sizeof(int));
	cudaMalloc((void**)&dv_out, n * sizeof(int));
	cudaMemcpy(idx, hs_idx, n * sizeof(int), cudaMemcpyHostToDevice);
	scatter << <1, n >> >(dv_out, dev_idata, f, idx);

	cudaMemcpy(odata, dv_out, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(f);
	cudaFree(dev_idata);
	cudaFree(idx);
	cudaFree(dv_out);
	free(hs_idx);
	free(hs_f);

	int count = 0;
	for (int i = 0; i < n; i++){
		if (odata[i] != 0){
			count++;
		}
	}
	return count;
}

}
}
