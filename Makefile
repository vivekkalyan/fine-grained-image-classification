DATA_PATH = data
RUNS_PATH = runs

install:
	 @[ "${VIRTUAL_ENV}" ] || ( echo ">> VIRTUAL_ENV is not set"; exit 1 )
	pip install -r requirements.txt

data:
	mkdir -p $(DATA_PATH)
	cd $(DATA_PATH); \
	curl -O http://imagenet.stanford.edu/internal/car196/cars_train.tgz; \
	curl -O http://imagenet.stanford.edu/internal/car196/cars_test.tgz; \
	curl -O https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz; \
	ls *.tgz | xargs -n1 tar -xzf
	rm $(DATA_PATH)/*.tgz
	cd $(DATA_PATH)/devkit; \
	curl -O http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat
	python process_data.py

runs:
	mkdir -p $(RUNS_PATH)

.PHONY: data
