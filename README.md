# gpgpu-sim_samples

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
