#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <stdint.h>

#define LOG						false
#define WRITE_FILES				true

#define PATH_AGGREGATION	4
#define	MAX_DISPARITY		128
#define CENSUS_WIDTH		9
#define CENSUS_HEIGHT		7

#define OCCLUDED_PIXEL		128
#define MISMATCHED_PIXEL	129

#define TOP				(CENSUS_HEIGHT-1)/2
#define LEFT			(CENSUS_WIDTH-1)/2

typedef uint32_t cost_t;
#define MAX_COST		30

#define BLOCK_SIZE					256
#define COSTAGG_BLOCKSIZE			GPU_THREADS_PER_BLOCK
#define COSTAGG_BLOCKSIZE_HORIZ		GPU_THREADS_PER_BLOCK

#define ABS_THRESH 3.0
#define REL_THRESH 0.05

#endif /* CONFIGURATION_H_ */
