# Reproduce every number and figure.  Usage: make all   (≈ 25 min on 8 cores)
PY ?= $(HOME)/.venvs/posterior-topology/bin/python

.PHONY: all test paper problem0 problem2 problem3 large_d atlas validate venv
all: test problem0 problem2 problem3 large_d atlas validate

venv:            ## venv lives OUTSIDE iCloud-synced Desktop (iCloud cancels reads inside it)
	python3 -m venv $(HOME)/.venvs/posterior-topology
	$(HOME)/.venvs/posterior-topology/bin/pip install -r requirements.txt
	ln -sf $(HOME)/.venvs/posterior-topology venv.nosync

test:     ; $(PY) -m pytest -q tests
problem0: ; $(PY) scripts/problem0.py
problem2: ; $(PY) scripts/problem2.py
problem3: ; $(PY) scripts/problem3.py
large_d:  ; $(PY) scripts/large_d.py
atlas:    ; $(PY) scripts/atlas.py
validate: ; $(PY) scripts/validate.py

paper:           ## every number, table and figure in paper/full/main.tex, then the PDF
	$(PY) scripts/table1.py
	$(PY) scripts/paper_numbers_check.py
	$(PY) scripts/problem0.py
	$(PY) scripts/kappa_next_order.py
	$(PY) scripts/exact_thresholds.py
	$(PY) scripts/validate_topology.py
	$(PY) scripts/inconsistent_data_check.py
	$(PY) scripts/inconsistent_data_check_fine.py
	$(PY) scripts/three_range_hole.py
	$(PY) scripts/three_range_smooth_max_search.py
	$(PY) scripts/three_range_fermat_example.py
	$(PY) scripts/three_range_fermat_hole.py
	$(PY) scripts/nranges_check.py
	$(PY) scripts/unequal_noise_check.py
	$(PY) scripts/unequal_noise_classification.py
	$(PY) scripts/unequal_certificate_check.py
	$(PY) scripts/large_d_next_order.py
	$(PY) scripts/sqrt_law_geometry_check.py
	$(PY) scripts/flip_sqrt_law.py
	$(PY) scripts/coverage.py
	$(PY) scripts/paper_figures.py
	cd paper/full && tectonic main.tex
