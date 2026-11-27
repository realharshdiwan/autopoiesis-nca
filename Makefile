PYTHON ?= $(shell if [ -f ../.venv/bin/python3 ]; then echo ../.venv/bin/python3; elif [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)
PYTEST ?= $(shell if [ -f ../.venv/bin/pytest ]; then echo ../.venv/bin/pytest; elif [ -f .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)

.PHONY: all test smoke train-all eval figures paper clean

all: test smoke

test:
	PYTHONPATH=. $(PYTEST) tests/test_sanity.py -v

smoke:
	PYTHONPATH=. $(PYTHON) scripts/run_ablations.py --smoke-test
	PYTHONPATH=. $(PYTHON) scripts/plot_figures.py

train-all:
	PYTHONPATH=. $(PYTHON) -c "import scripts.run_ablations; print('Training full 5-seed matrix...');"
	PYTHONPATH=. $(PYTHON) scripts/run_ablations.py

eval:
	PYTHONPATH=. $(PYTHON) scripts/run_ablations.py

figures:
	PYTHONPATH=. $(PYTHON) scripts/plot_figures.py

paper: figures
	@which tectonic >/dev/null 2>&1 && (cd paper && tectonic main.tex) || (which pdflatex >/dev/null 2>&1 && (cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)) || echo "LaTeX compiler not found on host, paper source generated in paper/main.tex"

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__ .pytest_cache
