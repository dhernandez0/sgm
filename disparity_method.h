#ifndef DISPARITY_METHOD_H_
#define DISPARITY_METHOD_H_

#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"
#include "debug.h"

void init_disparity_method(const uint8_t _p1, const uint8_t _p2);
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, const char* directory, const char* fname);
void finish_disparity_method();
static void free_memory();

#endif /* DISPARITY_METHOD_H_ */
