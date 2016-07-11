OBJS = dmsv sst_simple

all: $(OBJS)

$(OBJS):
	$(MAKE) -C $@

clean:
	for obj in $(OBJS); do $(MAKE) clean -C $$obj; done

.PHONY: $(OBJS)
