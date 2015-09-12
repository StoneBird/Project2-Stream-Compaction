#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	thrust::host_vector<int> hs_in(n);
	thrust::host_vector<int> hs_out(n);
	thrust::device_vector<int> dv_in(n);
	thrust::device_vector<int> dv_out(n);
	for (int i = 0; i < n; i++){
		hs_in[i] = idata[i];
	}
	dv_in = hs_in;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float msAdd = 0;
	cudaEventElapsedTime(&msAdd, start, stop);
	printf("Thrust scan: %f\n", msAdd);

	hs_out = dv_out;
	for (int i = 0; i < n; i++){
		odata[i] = hs_out[i];
	}
}

}
}
