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

#ifndef DEBUG_H_
#define DEBUG_H_

#include <iostream>
#include <stdio.h>
#include "configuration.h"

template<typename T>
void write_file(const char* fname, const T *data, const int size) {
	FILE* fp = fopen(fname, "wb");
	if (fp == NULL) {
		std::cerr << "Couldn't write transform file" << std::endl;
		exit(-1);
	}
	fwrite (data, sizeof(T), size, fp);
	fclose(fp);
}

void debug_log(const char *str);

#endif /* DEBUG_H_ */
