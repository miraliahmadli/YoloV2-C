# # CUDA
# cuda_lib.so: dnn_cuda.cu
# 	nvcc -Xcompiler -fPIC -shared -o $@ $^

# clean:
# 	rm -f cuda_lib.so

# CuBLAS
cublas_lib.so: dnn_cublas.cu
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

clean:
	rm -f cublas_lib.so
