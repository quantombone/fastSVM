all:    fastSVM

fastSVM: fastSVM.o
	nvcc --use_fast_math -O3 fastSVM.o -lcufft -lfreeimageplus -lboost_iostreams -o fastSVM

fastSVM.o: fastSVM.cu
	nvcc --use_fast_math -O3 -I /usr/local/cuda/sdk/common/inc -arch=sm_20 fastSVM.cu -c -o fastSVM.o

clean: 
	rm -rf fastSVM.o fastSVM


