#include "hamming_cost.h"

//d_transform0, d_transform1, d_cost, rows, cols
__global__ void
HammingDistanceCostKernel (  const cost_t *d_transform0, const cost_t *d_transform1,
		uint8_t *d_cost, const int rows, const int cols ) {
	//const int Dmax=   blockDim.x;  // Dmax is CTA size
	const int y=      blockIdx.x;  // y is CTA Identifier
	const int THRid = threadIdx.x; // THRid is Thread Identifier

	__shared__ cost_t SharedMatch[2*MAX_DISPARITY];
	__shared__ cost_t SharedBase [MAX_DISPARITY];

	SharedMatch [MAX_DISPARITY+THRid] = d_transform1[y*cols+0];  // init position

	int n_iter = cols/MAX_DISPARITY;
	for (int ix=0; ix<n_iter; ix++) {
		const int x = ix*MAX_DISPARITY;
		SharedMatch [THRid]      = SharedMatch [THRid + MAX_DISPARITY];
		SharedMatch [THRid+MAX_DISPARITY] = d_transform1 [y*cols+x+THRid];
		SharedBase  [THRid]      = d_transform0 [y*cols+x+THRid];

		__syncthreads();
		for (int i=0; i<MAX_DISPARITY; i++) {
			const cost_t base  = SharedBase [i];
			const cost_t match = SharedMatch[(MAX_DISPARITY-1-THRid)+1+i];
			d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = popcount( base ^ match );
		}
		__syncthreads();
	}
	// For images with cols not multiples of MAX_DISPARITY
	const int x = MAX_DISPARITY*(cols/MAX_DISPARITY);
	const int left = cols-x;
	if(left > 0) {
		SharedMatch [THRid]      = SharedMatch [THRid + MAX_DISPARITY];
		if(THRid < left) {
			SharedMatch [THRid+MAX_DISPARITY] = d_transform1 [y*cols+x+THRid];
			SharedBase  [THRid]      = d_transform0 [y*cols+x+THRid];
		}

		__syncthreads();
		for (int i=0; i<left; i++) {
			const cost_t base  = SharedBase [i];
			const cost_t match = SharedMatch[(MAX_DISPARITY-1-THRid)+1+i];
			d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = popcount( base ^ match );
		}
		__syncthreads();
	}
}
