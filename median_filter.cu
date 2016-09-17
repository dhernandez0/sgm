#include "median_filter.h"

__global__ void MedianFilter3x3(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {
	MedianFilter<3>(d_input, d_out, rows, cols);
}
