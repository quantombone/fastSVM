all:    fastSVM

fastSVM: fastSVM.o
	/usr/local/cuda/toolkit/bin/nvcc fastSVM.o -lboost_iostreams -lcufft -lfreeimageplus -o fastSVM

fastSVM.o: fastSVM.cu
	/usr/local/cuda/toolkit/bin/nvcc --use_fast_math -O3 -I /usr/local/cuda/sdk/common/inc -arch=sm_20 fastSVM.cu -c -o fastSVM.o

clean: 
	rm -rf fastSVM.o fastSVM


