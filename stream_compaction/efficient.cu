#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

	__global__ void scanUp(int d, int *idata){
		int k = threadIdx.x;
		if (k % (int)pow((double)2, (double)(d + 1)) == 0){
			idata[k - 1 + (int)pow((double)2, (double)(d + 1))] += idata[k - 1 + (int)pow((double)2, (double)d)];
		}
	}

	__global__ void scanDown(int d, int *idata){
		int k = threadIdx.x;
		if (k % (int)pow((double)2, (double)(d + 1)) == 0){
			int t = idata[k - 1 + (int)pow((double)2, (double)d)];
			idata[k - 1 + (int)pow((double)2, (double)d)] = idata[k - 1 + (int)pow((double)2, (double)(d + 1))];
			idata[k - 1 + (int)pow((double)2, (double)(d + 1))] += t;
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
    // TODO
    return -1;
}

}
}
