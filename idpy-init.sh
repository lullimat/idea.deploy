# Script for initializing idea.deploy python virtual environment
# Copyright (C) 2020-2025 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 10/6/2025

source .idpy-env

echo "Welcome to idea.deploy!"
echo

echo "Selecting PyPI/pythonhosted servers"
echo "Please select the region/servers for downloading python packages"
echo
for((SERVER_NAME_I=0; SERVER_NAME_I<${#PYHOSTED_SERVERS[@]}; SERVER_NAME_I++))
do
    echo "${SERVER_NAME_I}) ${PYHOSTED_SERVERS[${SERVER_NAME_I}]}/${PYPI_SERVERS[${SERVER_NAME_I}]}"
done
echo "$((${#PYPI_SERVERS[@]}))) for custom addresses"
echo

while true
do
    read -p "Enter selection (press 'return' for default 0): " SELECTED_SERVER
    if [ -z ${SELECTED_SERVER} ]
    then
        SELECTED_SERVER=0
    fi

    if((SELECTED_SERVER >= 0 && SELECTED_SERVER <= ${#PYPI_SERVERS[@]}))
    then
        break
    else
        echo
        echo "Please enter an integer between 0 and $((${#PYPI_SERVERS[@]}))"
        echo
    fi
done
echo

if((SELECTED_SERVER < ${#PYPI_SERVERS[@]}))
then
    USE_PYPI_SERVER=${PYPI_SERVERS[${SELECTED_SERVER}]}
    USE_PYHOSTED_SERVER=${PYHOSTED_SERVERS[${SELECTED_SERVER}]}
    PIP_SERVER_OPTION=${PIP_SERVER_OPTION_LIST[${SELECTED_SERVER}]}
elif((SELECTED_SERVER == ${#PYPI_SERVERS[@]}))
then
    echo "If you wish to add this choice as one of the default options you can modify"
    echo "the variables 'PYPI_SERVERS', 'PYHOSTED_SERVERS' and 'PIP_SERVER_OPTION_LIST'"
    echo "in the file '.idpy-env'"
    echo    
    echo "Please insert the servers addresses and options"
    read -p "Pythonhosted server: " USE_PYHOSTED_SERVER
    read -p "Pypi server: " USE_PYPI_SERVER
    read -p "Pip server options: " PIP_SERVER_OPTION
    if [ -z ${PIP_SERVER_OPTION} ]
    then
        PIP_SERVER_OPTION=""
    fi
fi
echo

ISTHERE_WGET=$(command -v wget >/dev/null 2>&1 && echo 1 || echo 0)
ISTHERE_CURL=$(command -v curl >/dev/null 2>&1 && echo 1 || echo 0)

# Never forget that ${VENV} is a global path

## Check if Python3 is installed
echo -n "Checking python3 installation:... "
## Python3
PY3_F=$(command -v python3 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p5_F=$(command -v python3.5 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p6_F=$(command -v python3.6 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p7_F=$(command -v python3.7 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p8_F=$(command -v python3.8 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p9_F=$(command -v python3.9 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p10_F=$(command -v python3.10 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p11_F=$(command -v python3.11 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p12_F=$(command -v python3.12 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p13_F=$(command -v python3.13 >/dev/null 2>&1 && echo 1 || echo 0)
##
if ((${PY3p13_F} && 0))
then
    ID_PYTHON=python3.13
    PY_PATH=$(which python3.13)
    echo "Found ${PY_PATH}"
elif ((${PY3p12_F} && 0))
then
    ID_PYTHON=python3.12
    PY_PATH=$(which python3.12)
    echo "Found ${PY_PATH}"
elif ((${PY3p11_F}))
then
    ID_PYTHON=python3.11
    PY_PATH=$(which python3.11)
    echo "Found ${PY_PATH}"
elif ((${PY3p10_F}))
then
    ID_PYTHON=python3.10
    PY_PATH=$(which python3.10)
    echo "Found ${PY_PATH}"
elif ((${PY3p9_F}))
then
    ID_PYTHON=python3.9
    PY_PATH=$(which python3.9)
    echo "Found ${PY_PATH}"
elif ((${PY3p8_F}))
then
    ID_PYTHON=python3.8
    PY_PATH=$(which python3.8)
    echo "Found ${PY_PATH}"
elif ((${PY3p7_F}))
then
    ID_PYTHON=python3.7
    PY_PATH=$(which python3.7)
    echo "Found ${PY_PATH}"
elif ((${PY3p6_F}))
then
    ID_PYTHON=python3.6
    PY_PATH=$(which python3.6)
    echo "Found ${PY_PATH}"
elif ((${PY3p5_F}))
then
    ID_PYTHON=python3.5
    PY_PATH=$(which python3.5)
    echo "Found ${PY_PATH}"
elif ((${PY3_F}))
then
    ID_PYTHON=python3
    PY_PATH=$(which python3)
    echo "Found ${PY_PATH}"
fi
# Ask the interpreter whether a binding is importable.
#
# This used to be `ls ${VENV_SITE_PKG} | grep ${DIR_PYOPENCL}`, but
# DIR_PYOPENCL and DIR_PYCUDA are defined nowhere in this repository. grep was
# therefore invoked with no pattern, printed its usage message to the terminal
# on every run, and always reported "not found" -- so the "Found!" branch was
# unreachable and the binding was reinstalled every time.
#
# Importing is also the question actually being asked. A directory in
# site-packages whose name happens to match is not the same as a binding that
# loads, and for pycuda in particular a broken build is exactly the case worth
# detecting.
#
# ${VENV_BIN}/python by explicit path: this runs before `source activate`.
IsImportable () {
    ${VENV_BIN}/python -c "import ${1}" >/dev/null 2>&1 && echo 1 || echo 0
}

## Check if virtual environment is installed
echo -n "Checking Python virtual environment..."
VENV_F=0
CHECK_VENV_F=0
if [ -d ${VENV_BIN} ]
then
   echo "Found!"
   VENV_F=1
   CHECK_VENV_F=1
else
    ## Setting up virtual environment
    echo "Not found!"
    echo "Setting up python virtual environment"
    CHECK_VENV_F=$(${PY_PATH} -m venv ${VENV} && echo 1 || echo 0)
fi

if((CHECK_VENV_F == 0))
then
    echo "There is some problem with the virtual environment installation"
    exit 1
fi

## Check if pyopencl is installed
echo -n "Checking pyopencl..."
PYOPENCL_F=$(IsImportable pyopencl)
if((PYOPENCL_F))
then
    echo "Found!"
else
    echo "Not found; pip will install it"
fi

#####################
## Checking CUDA
# echo -n "Looking for CUDA installation (for pyopencl headers):..."
echo -n "Checking CUDA installation..."
if [ -f ${VENV_ROOT}/cuda_paths ]
then
    CUDA_F=0
    while IFS= read -r line
    do
	if [ -d ${line} ]
	then
	    CUDA_F=1
	    CUDA_PATH=${line}
	fi
    done < ${VENV_ROOT}/cuda_paths
else
    echo "File ${VENV_ROOT}/cuda_paths not found!"
    echo "Looks like something is wrong with the installation"
    echo "Bye Bye!"
    exit 1
fi

if ((CUDA_F))
then
    echo "Found ${CUDA_PATH}"
    echo ${CUDA_PATH} > ${ID_CUDA_PATH_FOUND}
    CUDA_EXPORT_PATH_STRING="export PATH=\${PATH}:${CUDA_PATH}/bin"
    CUDA_EXPORT_LD_LIB_PATH_STRING="export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${CUDA_PATH}/lib:${CUDA_PATH}/lib64"
    CHECK_CUDA_PATH_ACTIVATE=$(grep "${CUDA_EXPORT_PATH_STRING}" \
        ${VENV_BIN}/activate &>/dev/null && echo 1 || echo 0)
    CHECK_CUDA_LD_PATH_ACTIVATE=$(grep "${CUDA_EXPORT_LD_LIB_PATH_STRING}" \
        ${VENV_BIN}/activate &>/dev/null && echo 1 || echo 0)

    if((CHECK_CUDA_PATH_ACTIVATE == 0))
    then
	echo "Appending line"
	echo "${CUDA_EXPORT_PATH_STRING}"
	echo "To 'activate' script: ${VENV_BIN}/activate"
	echo "${CUDA_EXPORT_PATH_STRING}" >> ${VENV_BIN}/activate
	echo
    fi
    if((CHECK_CUDA_LD_PATH_ACTIVATE == 0))
    then
	echo "Appending line"
	echo "${CUDA_EXPORT_LD_LIB_PATH_STRING}"
	echo "To 'activate' script: ${VENV_BIN}/activate"
	echo "${CUDA_EXPORT_LD_LIB_PATH_STRING}" >> ${VENV_BIN}/activate
	echo
    fi
else
    echo "Not Found"
    echo > ${ID_CUDA_PATH_FOUND}
fi

## Check if pycuda is installed
if((CUDA_F))
then
    echo -n "Checking pycuda..."
    PYCUDA_F=$(IsImportable pycuda)
    if((PYCUDA_F))
    then
	echo "Found!"
    else
	echo "Not found; pip will install it"
    fi
fi

#####################
## Checking OpenMpi
#####################
MPICC_F=$(which mpicc 1>/dev/null 2>/dev/null && echo 1 || echo 0)

#####################
## Checking Metal / pymetallic (macOS)
#####################
METAL_F=0
SWIFT_F=$(command -v swiftc >/dev/null 2>&1 && echo 1 || echo 0)
echo -n "Checking Metal/pymetallic prerequisites..."
if [[ "${IDEP_OS}" == "MacOS" ]]
then
    if ((SWIFT_F))
    then
        METAL_F=1
        echo "Found (Darwin + swiftc); will clone/patch/build via scripts/install-pymetallic.sh"
    else
        echo "Skipping"
        echo "  - swiftc not found (install Xcode / Swift toolchain for Metal)"
    fi
else
    echo "Skipping (Metal is macOS-only; OS=${IDEP_OS})"
fi

echo "Sourcing virtual environment"
source ${VENV}/bin/activate

if((VENV_F == 0))
then
    echo "Pip installing requirements"
    pip install --upgrade pip setuptools wheel ${PIP_SERVER_OPTION}
    pip install -r ${VENV_ROOT}/requirements.txt ${PIP_SERVER_OPTION}
    ## Install pycuda if cuda is found and it is not already importable.
    ## PYCUDA_F was computed but never consulted while its probe was broken;
    ## && short-circuits, so an unset PYCUDA_F (no CUDA) is harmless.
    if ((CUDA_F && PYCUDA_F == 0))
    then
    	pip install pycuda ${PIP_SERVER_OPTION}
    fi
    if((PYOPENCL_F == 0))
    then
        pip install pyopencl ${PIP_SERVER_OPTION}
    fi
    ## Install mpi4py if mpicc is found
    if((MPICC_F))
    then
    	pip install mpi4py ${PIP_SERVER_OPTION}
    else
	    echo "No MPI installation found (which mpicc did not return a path)"
    fi
    ## The classic-notebook extension setup that stood here is gone, and the
    ## features it installed are not.
    ##
    ## Eleven `jupyter contrib nbextension` / `jupyter nbextension` /
    ## `jupyter serverextension` / `jupyter labextension install` calls used to
    ## run here. Every one of them now fails: jupyter_contrib_nbextensions
    ## wants pkg_resources (removed in setuptools 81) and notebook.nbextensions
    ## (removed in notebook 7), the nbextension and serverextension
    ## subcommands no longer exist, and `labextension install` is a deprecated
    ## source build that demands nodejs >= 20.
    ##
    ## What they provided ships by default now, so nothing was lost:
    ##   toc2/main         -> table of contents, built into JupyterLab 4 and
    ##                        notebook 7 (the papers' READMEs still describe
    ##                        reading the notebooks with it)
    ##   codefolding/main  -> built into the CodeMirror 6 editor
    ##   widgetsnbextension,
    ##   jupyter-matplotlib -> prebuilt extensions, installed by pip alone
    ##                        (ipywidgets and ipympl are in requirements.txt)
    ##
    ## Adding the virtual environment to jupyter. `python`, not ${ID_PYTHON}:
    ## the venv is active here, whereas ID_PYTHON is a bare command name
    ## resolved against PATH.
    python -m ipykernel install --name idpy-env --display-name "idea.deploy" --user
    ## --env ${CUDA_EXPORT}
fi

## Install idpy itself, on a new AND an existing virtual environment.
##
## This sits outside the block above deliberately. It used to be inside it, so
## it ran only when the venv was created -- which means a machine whose
## py-env/idpy-env predates the Phase 0b restructure, re-running this script
## after pulling it, would get every dependency and not idpy, and then fail at
## `import idpy` for no visible reason. Under the old flat layout that worked
## by accident, because the repository root was importable from the cwd.
##
## Editable, because this environment exists to develop the tree it was built
## from, and idempotent, so running it on a fresh venv costs nothing.
pip install -e . ${PIP_SERVER_OPTION}

## Build/install local pymetallic for Metal (new and existing venvs)
if ((METAL_F))
then
    bash scripts/install-pymetallic.sh || echo "WARNING: scripts/install-pymetallic.sh failed; Metal backend will be unavailable."
fi

echo
echo "Running the idpy.test suite to check the installation"

## Dotted module name, not a path: as of Phase 0b the package lives at
## src/idpy and is reached through the install above, not through the cwd.
TEST_RES=$(python -m unittest -v idpy.test 1>${IDPY_TEST_STDOUT} 2>${IDPY_TEST_STDERR} && echo 1 || echo 0)
echo "Python tests have..."
if((TEST_RES == 1))
then
    echo "PASSED!"
else
    echo "FAILED! Please, check ${IDPY_TEST_STDERR} and run a more detailed testing..."
fi
echo
echo
## ALIASES
##
## Both blocks below are outside the "new virtual environment" guard, so they
## run on every invocation. They used to append unconditionally: a second run
## wrote all seven aliases again and added a second `source` line to .bashrc,
## a third run a third, and so on.
##
## Truncate and rewrite rather than test-and-append -- idempotent by
## construction, and it picks up any alias whose definition has changed in
## .idpy-env since the last run.
: > ${VENV_ALIASES}
for((ALIAS_I=0; ALIAS_I<${#IDPY_ALIASES[@]}; ALIAS_I++))
do
	echo "${IDPY_ALIASES[ALIAS_I]}" >> ${VENV_ALIASES}
done
## Appending lines at the end of the .bashrc to source the aliases, once.
## .bashrc belongs to the user and may hold anything, so this one is
## test-and-append: the banner is the marker.
if ! grep -qF "${ID_BASHRC_BANNER}" "${HOME}/.bashrc" 2>/dev/null
then
    echo "${ID_BASHRC_BANNER}" >> ${HOME}/.bashrc
    echo "${ID_BASHRC_ALIASES_OPT}" >> ${HOME}/.bashrc
    echo "${ID_BASHRC_SOURCE_ALIASES}" >> ${HOME}/.bashrc
fi

echo "For using the alias in the present shell: source ${HOME}/.bashrc"

echo "Python virtual environment for idea.deploy initialized!"
echo
