#ifndef HAMMING_COST_H_
#define HAMMING_COST_H_

#include "configuration.h"
#include "util.h"
#include <stdint.h>

__global__ void
HammingDistanceCostKernel (  const cost_t *d_transform0, const cost_t *d_transform1,
		uint8_t *d_cost, const int rows, const int cols );

#endif /* HAMMING_COST_H_ */
