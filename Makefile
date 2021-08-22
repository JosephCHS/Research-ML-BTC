# Minimal makefile for Sphinx documentation
#

all: init run clean

init:
	apt-get install python-setuptools
	sudo easy_install pip==21.1.1
	sudo pip install virtualenv
	python3 -m venv .venv
	./.venv/bin/activate
	pip install -r ./source/requirements.txt

run: .venv/bin/activate
	./.venv/bin/python3 ./source/main.py

clean:
	rm -rf .venv
	rm -rf build
	rm -rf documentation/*


.PHONY: all init run clean Makefile
