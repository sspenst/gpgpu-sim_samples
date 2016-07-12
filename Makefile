OBJS = dmsv smsv sst_simple

all: $(OBJS)

$(OBJS):
	@echo "Compiling $@..."
	@$(MAKE) -s -C $@

clean:
	@for obj in $(OBJS); do $(MAKE) clean -s -C $$obj; done

.PHONY: $(OBJS)
