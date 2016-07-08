# gpgpu-sim_samples

Run this to compile the CUDA file:
nvcc -ccbin g++ -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=compute_20 -lcudart -o sst_simple sst_simple.cu

774:
Stores 8 floats in sstarr memory simulatenously using the SST instruction

775:
Stores an array in sstarr memory, then retrieves it in reverse order

777:
Squeezes out the zeros from an array and updates the original array with the result

778:
Same as 777 except now the instruction returns the end address of the sparse array

779:
Indices are now stored corresponding to values. SST now returns the number of elements instead of the device memory address

781:
Much cleaner implementation of the inline assembly for the SST instruction. Also got rid of unnecessary variables
