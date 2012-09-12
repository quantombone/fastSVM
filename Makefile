all:    fastSVM

fastSVM: fastSVM.o
	nvcc fastSVM.o -lcufft -lfreeimageplus -lboost_iostreams -o fastSVM

fastSVM.o: fastSVM.cu
	nvcc -g -G -I /usr/local/cuda/sdk/common/inc -arch=sm_20 fastSVM.cu -c -o fastSVM.o

clean: 
	rm -rf fastSVM.o fastSVM


