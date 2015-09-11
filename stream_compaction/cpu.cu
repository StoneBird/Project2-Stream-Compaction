#include <cstdio>
#include "cpu.h"
#include <stdlib.h>

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (exclusive prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
	odata[0] = 0;
	for (int i = 1; i < n; i++){
		odata[i] = odata[i - 1] + idata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int count = 0;
	int j = 0;
	for (int i = 0; i < n; i++){
		if (idata[i] != 0){
			count++;
			odata[j] = idata[i];
			j++;
		}
	}
    return count;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *c;
	c = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++){
		if (idata[i] != 0){
			c[i] = 1;
		}
		else {
			c[i] = 0;
		}
	}
	int *d, *e;
	d = (int *)malloc(n * sizeof(int));
	scan(n, d, c);
	for (int i = 0; i < n; i++){
		if (c[i] == 1){
			odata[d[i]] = idata[i];
		}
	}
	free(c);
	free(d);
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
