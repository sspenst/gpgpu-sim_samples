SM = 20
OBJS = gold-basic gold

all: data $(OBJS) clc

data: data.py
	python $^

$(OBJS):
	gcc -g -Wall $@.c -o $@ -std=c99

clc: clc.cu
	nvcc -ccbin g++ -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM) -lcudart -o clc clc.cu

clean: 
	rm -f $(OBJS) input clc

.PHONY: $(OBJS)
