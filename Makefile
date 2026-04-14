.PHONY: Startup install all clean

env:
	python -m venv .venv

install:
	.venv\Scripts\pip install -r requirements.txt

all: Startup install

clean:
	rmdir /s /q .venv
