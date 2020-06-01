# CUDA
cuda_lib.so: dnn_cuda.cu
	nvcc -Xcompiler -fPIC -shared -o $@ $^

clean:
	rm -f cuda_lib.so
