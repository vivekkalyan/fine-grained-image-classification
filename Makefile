DATA_PATH = data

install:
	ifndef VIRTUAL_ENV
	$(error VIRTUAL_ENV is not set)
	endif
	pip install -r requirements.txt

data:
	mkdir -p $(DATA_PATH)

.PHONY: data
