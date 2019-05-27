DATA_PATH = data

install:
	ifndef VIRTUAL_ENV
	$(error VIRTUAL_ENV is not set)
	endif
	pip install -r requirements.txt

data:
	mkdir -p $(DATA_PATH)
	(cd $(DATA_PATH) && curl -O http://imagenet.stanford.edu/internal/car196/cars_train.tgz)
	(cd $(DATA_PATH) && curl -O http://imagenet.stanford.edu/internal/car196/cars_test.tgz)
	(cd $(DATA_PATH) && curl -O https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)
	(cd $(DATA_PATH) && ls *.tgz | xargs -n1 tar -xzf)
	rm $(DATA_PATH)/*.tgz

.PHONY: data
