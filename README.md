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
@inproceedings{sgm_gpu_iccs2016,
  author    = {Daniel Hernandez-Juarez and
               Alejandro Chac{\'{o}}n and
               Antonio Espinosa and
               David V{\'{a}}zquez and
               Juan Carlos Moure and
               Antonio M. L{\'{o}}pez},
  title     = {Embedded Real-time Stereo Estimation via Semi-Global Matching on the
               {GPU}},
  booktitle = {International Conference on Computational Science 2016, {ICCS} 2016,
               6-8 June 2016, San Diego, California, {USA}},
  pages     = {143--153},
  year      = {2016},
  crossref  = {DBLP:conf/iccS/2016},
  url       = {http://dx.doi.org/10.1016/j.procs.2016.05.305},
  doi       = {10.1016/j.procs.2016.05.305},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/iccS/JuarezCEVML16},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
