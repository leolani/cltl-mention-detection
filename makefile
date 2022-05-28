SHELL = /bin/bash

project_dependencies ?= $(addprefix $(project_root)/, emissor \
    cltl-combot \
    cltl-knowledgerepresentation \
    cltl-requirements \
    cltl-face-recognition \
    cltl-object-recognition)

git_remote ?= https://github.com/leolani

include util/make/makefile.base.mk
include util/make/makefile.component.mk
include util/make/makefile.py.base.mk
include util/make/makefile.git.mk

.PHONY: build
build: spacy.lock


spacy.lock:
	source venv/bin/activate; python -m spacy download en
	touch spacy.lock


.PHONY: clean
clean:
	rm -f spacy.lock