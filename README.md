# Semi-Global Matching on the GPU

This is the implementation of [Embedded real-time stereo estimation via Semi-Global Matching on the GPU](http://www.sciencedirect.com/science/article/pii/S1877050916306561), [D. Hernandez-Juarez](http://www.cvc.uab.es/people/dhernandez/) et al, ICCS 2016.

## How to compile and test

Simply use CMake and target the output directory as "build". In command line this would be (from the project root folder):

```
mkdir build
cd build
cmake ..
make
```

## How to use it

Type: `./sgm dir p1 p2`

The arguments `p1` and `p2` are semi-global matching parameters, for more information read the SGM paper.

`dir` is the name of the directory which needs this format:

```
dir
---- left (images taken from the left camera)
---- right (right camera)
---- disparities (results will be here)
```

## Related Publications

[Embedded real-time stereo estimation via Semi-Global Matching on the GPU](http://www.sciencedirect.com/science/article/pii/S1877050916306561)
[D. Hernandez-Juarez](http://www.cvc.uab.es/people/dhernandez/), A. Chacón, A. Espinosa, D. Vázquez, J. C. Moure, and A. M. López
ICCS2016 – International Conference on Computational Science 2016

## Requirements

- OpenCV
- CUDA
- CMake

## Limitations

- Maximum disparity has to be 128
- Image width and height must be a divisible by 4

## What to cite

If you use this code for your research, please kindly cite:

```
@article{sgm_gpu_iccs2016,
author = "D. Hernandez-Juarez and A. Chacon and A. Espinosa and D. Vazquez and J.C. Moure and A.M. Lopez",
title = "Embedded Real-time Stereo Estimation via Semi-global Matching on the GPU ",
journal = "Procedia Computer Science ",
volume = "80",
number = "",
pages = "143 - 153",
note = "International Conference on Computational Science (ICCS), 6-8 June 2016, San Diego, California ",
year         = {2016}
}
```
