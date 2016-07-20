OBJS = dmdv dmsv smsv sst_simple stripes

all: $(OBJS)

$(OBJS):
	@echo "Compiling $@..."
	@$(MAKE) -s -C $@

clean:
	@for obj in $(OBJS); do $(MAKE) clean -s -C $$obj; done

.PHONY: $(OBJS)
