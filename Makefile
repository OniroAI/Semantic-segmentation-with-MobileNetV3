NAME=person_segmentation_tf

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all stop build run

all: stop build run

build:
	docker build -t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

run:
	docker run --rm -it --shm-size=16g \
                $(GPUS_OPTION) \
		--net=host \
		--runtime=nvidia \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		bash
