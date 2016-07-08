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

781:
Much cleaner implementation of the inline assembly for the SST instruction. Also got rid of unnecessary variables

782:
You can now change LENGTH without crashing GPGPU-Sim. Also reduced the number of lines of assembly and added some more output data
