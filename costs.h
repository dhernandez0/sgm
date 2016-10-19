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

#ifndef COSTS_H_
#define COSTS_H_

#include <stdint.h>
#include "configuration.h"

__global__ void CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols);

#endif /* COSTS_H_ */
