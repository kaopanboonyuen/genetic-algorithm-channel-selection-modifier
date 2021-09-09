# genetic algorithm channel selection v.modifer

This hub contains the modifier program of Sweta Shaw to perform channel selection for hyperspectral images.

### How to convert a .tif file into a .mat file?

tmp_img = imread('Example.tiff');
save('myTiff2mat','tmp_img');

** where 'myTiff2mat' is file name you want to be e.g. change name to 'supernitza555'

### genetic_function.py
  - This is a utility file. It contains all the functions required to implement the genetic algorithm for band selection for hyperspectral images

### main.py
  - This file contains the driver code for running the program.
  
- In this program we are selecting 15 bands out of 170 bands using genetic algorithm. 
- The population consists of 8 individuals
- Each individual has 10 genes.

For running the program run - 

```sh
$ python main.py
```
