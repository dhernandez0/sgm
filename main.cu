/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include <numeric>
#include <stdlib.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include "disparity_method.h"

bool directory_exists(const char* dir) {
	DIR* d = opendir(dir);
	bool ok = false;
	if(d) {
	    closedir(d);
	    ok = true;
	}
	return ok;
}

void disparity_errors(cv::Mat estimation, const char* gt_file, int *n, int *n_err) {
	int nlocal = 0;
	int nerrlocal = 0;

	cv::Mat gt_image = cv::imread(gt_file, cv::IMREAD_UNCHANGED);
	if(!gt_image.data) {
		std::cerr << "Couldn't read the file " << gt_file << std::endl;
		exit(EXIT_FAILURE);
	}
	if(estimation.rows != gt_image.rows || estimation.cols != gt_image.cols) {
		std::cerr << "Ground truth must have the same dimesions" << std::endl;
		exit(EXIT_FAILURE);
	}
	const int type = estimation.type();
	const uchar depth = type & CV_MAT_DEPTH_MASK;
	for(int i = 0; i < gt_image.rows; i++) {
		for(int j = 0; j < gt_image.cols; j++) {
			const uint16_t gt = gt_image.at<uint16_t>(i, j);
			if(gt > 0) {
				const float gt_f = ((float)gt)/256.0f;
				float est;
				if(depth == CV_8U) {
					est = (float) estimation.at<uint8_t>(i, j);
				} else {
					est = estimation.at<float>(i, j);
				}
				const float err = fabsf(est-gt_f);
				const float ratio = err/fabsf(gt_f);
				if(err > ABS_THRESH && ratio > REL_THRESH) {
					nerrlocal++;
				}
				nlocal++;
			}
		}
	}
	*n += nlocal;
	*n_err += nerrlocal;
}

bool check_directories_exist(const char* directory, const char* left_dir, const char* right_dir, const char* disparity_dir) {
	char left_dir_sub[PATH_MAX];
	char right_dir_sub[PATH_MAX];
	char disparity_dir_sub[PATH_MAX];
	sprintf(left_dir_sub, "%s/%s", directory, left_dir);
	sprintf(right_dir_sub, "%s/%s", directory, right_dir);
	sprintf(disparity_dir_sub, "%s/%s", directory, disparity_dir);

	return directory_exists(left_dir_sub) && directory_exists(right_dir_sub) && directory_exists(disparity_dir_sub);
}

int main(int argc, char *argv[]) {
	if(argc < 4) {
		std::cerr << "Usage: cuda_sgm dir p1 p2" << std::endl;
		return -1;
	}
	if(MAX_DISPARITY != 128) {
		std::cerr << "Due to implementation limitations MAX_DISPARITY must be 128" << std::endl;
		return -1;
	}
	if(PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) {
                std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
                return -1;
        }
	const char* directory = argv[1];
	uint8_t p1, p2;
	p1 = atoi(argv[2]);
	p2 = atoi(argv[3]);

	DIR *dp;
	struct dirent *ep;

	// Directories
	const char* left_dir = "left";
	const char* disparity_dir = "disparities";
	const char* right_dir = "right";
	const char* gt_dir = "gt";

	if(!check_directories_exist(directory, left_dir, right_dir, disparity_dir)) {
		std::cerr << "We need <left>, <right> and <disparities> directories" << std::endl;
		exit(EXIT_FAILURE);
	}
	char abs_left_dir[PATH_MAX];
    sprintf(abs_left_dir, "%s/%s", directory, left_dir);
	dp = opendir(abs_left_dir);
	if (dp == NULL) {
		std::cerr << "Invalid directory: " << abs_left_dir << std::endl;
		exit(EXIT_FAILURE);
	}
	char left_file[PATH_MAX];
	char right_file[PATH_MAX];
	char dis_file[PATH_MAX];
	char gt_file[PATH_MAX];
	char gt_dir_sub[PATH_MAX];

	sprintf(gt_dir_sub, "%s/%s", directory, gt_dir);
	const bool has_gt = directory_exists(gt_dir_sub);
	int n = 0;
	int n_err = 0;
	std::vector<float> times;

	init_disparity_method(p1, p2);
	while ((ep = readdir(dp)) != NULL) {
		// Skip directories
		if (!strcmp (ep->d_name, "."))
			continue;
		if (!strcmp (ep->d_name, ".."))
			continue;

		sprintf(left_file, "%s/%s/%s", directory, left_dir, ep->d_name);
		sprintf(right_file, "%s/%s/%s", directory, right_dir, ep->d_name);
		sprintf(dis_file, "%s/%s/%s", directory, disparity_dir, ep->d_name);
		sprintf(gt_file, "%s/%s/%s", directory, gt_dir, ep->d_name);
		int gt_len = strlen(gt_file);

		cv::Mat h_im0 = cv::imread(left_file);
		if(!h_im0.data) {
			std::cerr << "Couldn't read the file " << left_file << std::endl;
			return EXIT_FAILURE;
		}
		cv::Mat h_im1 = cv::imread(right_file);
		if(!h_im1.data) {
			std::cerr << "Couldn't read the file " << right_file << std::endl;
			return EXIT_FAILURE;
		}

		// Convert images to grayscale
		if (h_im0.channels()>1) {
			cv::cvtColor(h_im0, h_im0, cv::COLOR_RGB2GRAY);
		}

		if (h_im1.channels()>1) {
			cv::cvtColor(h_im1, h_im1, cv::COLOR_RGB2GRAY);
		}

		if(h_im0.rows != h_im1.rows || h_im0.cols != h_im1.cols) {
			std::cerr << "Both images must have the same dimensions" << std::endl;
			return EXIT_FAILURE;
		}
		if(h_im0.rows % 4 != 0 || h_im0.cols % 4 != 0) {
                        std::cerr << "Due to implementation limitations image width and height must be a divisible by 4" << std::endl;
                        return EXIT_FAILURE;
		}

#if LOG
		std::cout << "processing: " << left_file << std::endl;
#endif
		// Compute
		float elapsed_time_ms;
		cv::Mat disparity_im = compute_disparity_method(h_im0, h_im1, &elapsed_time_ms, directory, ep->d_name);
#if LOG
		std::cout << "done" << std::endl;
#endif
		times.push_back(elapsed_time_ms);

		if(has_gt) {
			disparity_errors(disparity_im, gt_file, &n, &n_err);
		}
#if WRITE_FILES
	const int type = disparity_im.type();
	const uchar depth = type & CV_MAT_DEPTH_MASK;
	if(depth == CV_8U) {
		cv::imwrite(dis_file, disparity_im);
	} else {
		cv::Mat disparity16(disparity_im.rows, disparity_im.cols, CV_16UC1);
		for(int i = 0; i < disparity_im.rows; i++) {
			for(int j = 0; j < disparity_im.cols; j++) {
				const float d = disparity_im.at<float>(i, j)*256.0f;
				disparity16.at<uint16_t>(i, j) = (uint16_t) d;
			}
		}
		cv::imwrite(dis_file, disparity16);
	}
#endif
	}
	closedir(dp);
	finish_disparity_method();

	double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
	if(has_gt) {
		printf("%f\n", (float) n_err/n);
	} else {
		std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
	}

	return 0;
}
