ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))


SRCDIR=${ROOTDIR}/raw_emg_aug_tsgm
TESTDIR=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov_p
COVERAGERC=${ROOTDIR}/.coveragerc
INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv
CONDA_SUBDIR=${ROOTDIR}/cvenv
COVERAGERC=${ROOTDIR}/.coveragerc
DOCS_DIR=${ROOTDIR}/docs
TOXDIR=${ROOTDIR}/.tox
TOXJSON=${ROOTDIR}/tox.json
STATICDIR=${ROOTDIR}/static_analysis
LINTFILE=${STATICDIR}/lint.json
FLAKE8FILE=${STATICDIR}/flake8.log
MYPYFILE=${STATICDIR}/mypy.log

COVERAGE = coverage
UNITTEST_PARALLEL = unittest-parallel
PDOC= pdoc3
PYTHON=python
SYSPYTHON=python
PIP=pip
TOX=tox
PYTEST=pytest
PYLINT= pylint
FLAKE8= flake8
MYPY= mypy
CONDA=conda
CONDA_EXISTS := $(shell command -v conda 2> /dev/null)
# --system-site-packages for using system installed packages ex. from FreeBSD. 
VENV_OPTIONS=
VENV_TYPE ?= venv

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log

TOX_CORES=auto
BACKEND=current_tf

ifeq ($(VENV_TYPE), conda)
	ACTIVATE:= ${CONDA} activate ${CONDA_SUBDIR}
else
ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif
endif

.PHONY: all clean test docs

clean: clean_pypackages clean_venv clean_tox clean_cvenv
	@echo "Cleaning up build artifacts, virtual environments, and test logs..."

clean_pypackages:
	rm -rf pypackages

clean_venv:
	rm -rf ${VENV_SUBDIR}

clean_cvenv:
	rm -rf ${CONDA_SUBDIR}

clean_tox:
	rm -rf ${TOXDIR}
venv:
	${SYSPYTHON} -m venv --upgrade-deps ${VENV_OPTIONS} ${VENV_SUBDIR}
	${ACTIVATE}; ${PYTHON} -m ${PIP} install wheel setuptools pypackages

cvenv:
	@command -v conda >/dev/null 2>&1 || { echo "No conda → skip"; exit 0; }
	@echo "Creating conda env"
	${CONDA} create --prefix ${CONDA_SUBDIR}
	

pypackages: venv cvenv
	${ACTIVATE}; ${PYTHON} -m ${PIP} install -e ${ROOTDIR}[dev,${BACKEND}] --prefer-binary --log ${INSTALL_LOG_FILE}
	touch $@

test: pypackages
	mkdir -p ${LOGDIR}  
	${ACTIVATE}; ${COVERAGE} run --branch  --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} 2>&1 |tee -a ${LOGFILE}
	#${ACTIVATE}; ${COVERAGE} html --show-contexts


test_parallel: pypackages
	mkdir -p ${COVDIR} ${LOGDIR}
	${ACTIVATE}; ${UNITTEST_PARALLEL} --class-fixtures -v -t ${ROOTDIR} -s ${TESTDIR} -p '*_test.py' --coverage --coverage-rcfile ./.coveragerc --coverage-source ${SRCDIR} --coverage-html ${COVDIR}  2>&1 |tee -a ${LOGFILE}

docs: pypackages
	${ACTIVATE}; $(PDOC) --force --html ${SRCDIR} --output-dir ${DOCS_DIR}

profile: pypackages
	
	${ACTIVATE}; ${PYTEST} -n auto --cov-report=html --cov=${SRCDIR} --profile ${TESTDIR}

tox_check: pypackages
	${ACTIVATE}; ${TOX} -vv  -p ${TOX_CORES}  --result-json ${TOXJSON}

${STATICDIR}:
	mkdir -p ${STATICDIR}
flake8: pypackages ${STATICDIR}
	${ACTIVATE}; ${FLAKE8} --jobs auto ${SRCDIR} > ${FLAKE8FILE} || true

mypy: pypackages ${STATICDIR}
	${ACTIVATE}; ${MYPY} --pretty --show-error-context ${SRCDIR} > ${MYPYFILE} || true
lint: pypackages ${STATICDIR}
	${ACTIVATE}; ${PYLINT} -j 0 ${SRCDIR} --output-format=json > ${LINTFILE} || true

static_check: flake8 mypy lint

verify_installation: pypackages
	echo "Package versions"
	${ACTIVATE}; ${PYTHON} -c "import tensorflow as tf; import keras ; print(tf.__version__); print(keras.__version__)"
	echo "GPUs"
	${ACTIVATE}; ${PYTHON} -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"