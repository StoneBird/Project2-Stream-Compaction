#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <stdlib.h>

namespace StreamCompaction {
namespace Naive {

	__global__ void scanCol(int d, int *idata){
		int k = threadIdx.x;
		if (k >= (int)pow((double)2, (double)(d-1))){
			idata[k] = idata[k - (int)pow((double)2, (double)(d - 1))] + idata[k];
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
	for (int d = 1; d <= ilog2ceil(m); d++){
		scanCol<<<1, m>>>(d, dev_pidata);
	}
	cudaMemcpy(pidata, dev_pidata, m*sizeof(int), cudaMemcpyDeviceToHost);
	odata[0] = 0;
	for (int i = 1; i < n; i++){
		odata[i] = pidata[i-1];
	}
	cudaFree(dev_pidata);
	free(pidata);
}

}
}
