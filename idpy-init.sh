# Script for initializing idea.deploy python virtual environment
# Copyright (C) 2020-2021 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 25/6/2021

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

WGET_PYOPENCL_2019=https://${USE_PYHOSTED_SERVER}/packages/1b/0e/f49c0507610aae0bc2aba6ad1e79f87992d9e74e6ea55af23e436075502e/pyopencl-2019.1.tar.gz
WGET_PYOPENCL_2020=https://${USE_PYHOSTED_SERVER}/packages/a1/b5/c32aaa78e76fefcb294f4ad6aba7ec592d59b72356ca95bcc4abfb98af3e/pyopencl-2020.2.tar.gz
WGET_PYOPENCL_2021=https://${USE_PYHOSTED_SERVER}/packages/71/2f/e5c0860f86f8ea8d8044db7b661fccb954c200308d94d982352592eb88ee/pyopencl-2021.1.2.tar.gz
WGET_PYOPENCL_2022=https://${USE_PYHOSTED_SERVER}/packages/dd/4c/7f60601cb3e55d21cff151da2121d7d19e8f9f69e3519dc8fca2593668b2/pyopencl-2022.3.1.tar.gz
WGET_PYOPENCL=${WGET_PYOPENCL_2021}

WGET_PYCUDA_2019=https://${USE_PYHOSTED_SERVER}/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
WGET_PYCUDA_2020=https://${USE_PYHOSTED_SERVER}/packages/46/61/47d3235a4c13eec5a5f03594ddb268f4858734e02980afbcd806e6242fa5/pycuda-2020.1.tar.gz
WGET_PYCUDA_2021=https://${USE_PYHOSTED_SERVER}/packages/5a/56/4682a5118a234d15aa1c8768a528aac4858c7b04d2674e18d586d3dfda04/pycuda-2021.1.tar.gz
WGET_PYCUDA_2022=https://${USE_PYHOSTED_SERVER}/packages/78/09/9df5358ffb74d225243b56a65ffe196de481fcd8f731f55e41f2d5d36015/pycuda-2022.2.2.tar.gz
WGET_PYCUDA=${WGET_PYCUDA_2022}

TAR_PYOPENCL=$(echo ${WGET_PYOPENCL} | tr '/' ' ' | awk '{print($NF)}')
DIR_PYOPENCL=${TAR_PYOPENCL:0:${#TAR_PYOPENCL} - 7}
TAR_PYCUDA=$(echo ${WGET_PYCUDA} | tr '/' ' ' | awk '{print($NF)}')
DIR_PYCUDA=${TAR_PYCUDA:0:${#TAR_PYCUDA} - 7}

ISTHERE_WGET=$(command -v wget >/dev/null 2>&1 && echo 1 || echo 0)
ISTHERE_CURL=$(command -v curl >/dev/null 2>&1 && echo 1 || echo 0)

# Script scope
# Load local python env if None

# install requirements

# download pyopencl

# configure and build and install

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

##
if ((${PY3p10_F}))
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
# Set site-packages path
VENV_SITE_PKG=${VENV_LIB}/${ID_PYTHON}/site-packages/

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
echo -n "Looking for local pyopencl build..."
PYOPENCL_F=$(ls ${VENV_SITE_PKG} | grep ${DIR_PYOPENCL} 1>/dev/null && echo 1 || echo 0)
if((PYOPENCL_F))
then
    echo "Found!"
else
    echo "Not Found...need to build!"
fi

#####################
## Checking CUDA
echo -n "Looking for CUDA installation (for pyopencl headers):..."
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
    echo > ${VENV_ROOT}/cuda_path_found
fi

## Check if pycuda is installed
if((CUDA_F))
then
    echo -n "Looking for local pycuda build..."
    PYCUDA_F=$(ls ${VENV_SITE_PKG} | grep ${DIR_PYCUDA} 1>/dev/null && echo 1 || echo 0)
    if((PYCUDA_F))
    then
	echo "Found!"
    else
	echo "Not Found...need to build!"
    fi
fi

#####################
## Checking OpenMpi
#####################
MPICC_F=$(which mpicc 1>/dev/null 2>/dev/null && echo 1 || echo 0)

echo "Sourcing virtual environment"
source ${VENV}/bin/activate

if((VENV_F == 0))
then
    echo "Pip installing requirements"
    pip install --upgrade pip setuptools wheel ${PIP_SERVER_OPTION}
    pip install -r ${VENV_ROOT}/requirements.txt ${PIP_SERVER_OPTION}
    ## Install pycuda if cuda is found
    if ((CUDA_F && 0))
    then
	pip install pycuda ${PIP_SERVER_OPTION}
    fi
    ## Install mpi4py if mpicc is found
    if((MPICC_F))
    then
	pip install mpi4py ${PIP_SERVER_OPTION}
    else
	echo "No MPI installation found (which mpicc did not return a path)"
    fi
    ## Jupyter nbextension
    jupyter contrib nbextension install --user
    jupyter nbextensions_configurator enable --user
    ## Jupyter Folding Options
    jupyter nbextension enable codefolding/main
    ## Jupyter Sections Management
    jupyter nbextension enable toc2/main
    ## Further step to avoid toc and fodling disappear
    ## https://github.com/jupyter/help/issues/186
    jupyter nbextension enable --py widgetsnbextension
    ## Adding ipyparallel
    jupyter serverextension enable --py ipyparallel
    jupyter nbextension install --py ipyparallel
    jupyter nbextension enable --py ipyparallel

    pip install --upgrade jupyterlab
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter labextension install jupyter-matplotlib
    jupyter nbextension enable --py widgetsnbextension
    
    ## Adding virtual environemtn to jupyter
    ${ID_PYTHON} -m ipykernel install --name idpy-env --display-name "idea.deploy" --user
    ## --env ${CUDA_EXPORT}
fi

## Building pycuda
if((CUDA_F && PYCUDA_F == 0))
then
    if [ ! -f ${VENV_SRC}/${TAR_PYCUDA} ]
    then
	echo "Downloading pycuda source:..."
	if((ISTHERE_WGET))
	then
	    wget -P ${VENV_SRC} ${WGET_PYCUDA} --no-check-certificate
	elif((ISTHERE_CURL))
	then
	    mkdir ${VENV_SRC}
	    curl -o ${VENV_SRC}/${TAR_PYCUDA} ${WGET_PYCUDA}
	fi
	echo "Done (Downloading pycuda source)"
    fi
    
    echo "Configuring pycuda"
    STARTDIR=${PWD}
    cd ${VENV_SRC}
    tar zxfk ${TAR_PYCUDA}
    cd ${DIR_PYCUDA}
    ${ID_PYTHON} configure.py

    ## If CUDA is found I need to modify the files
    ## Thanks to: https://wiki.tiker.net/PyOpenCL/Installation/Windows/
    
    CUDA_PATH_SED=$(echo "${CUDA_PATH}" | sed 's/\//\\\//g')
    SED_STR_1_OLD="CUDA\_ROOT\ =\ '.*'"
    SED_STR_1_NEW="CUDA\_ROOT\ =\ '${CUDA_PATH_SED}'"
    
    echo ${SED_STR_1_NEW}
    
    sed -i.bkp "s/${SED_STR_1_OLD}/${SED_STR_1_NEW}/g" siteconf.py
    cat siteconf.py
    
    echo "Building pycuda"
    ${ID_PYTHON} setup.py build
    
    echo "Install pycuda"
    ${ID_PYTHON} setup.py install
    
    cd ${STARTDIR}
fi
echo
echo
## Building pyopencl
if((PYOPENCL_F == 0))
then
    if [ ! -f ${VENV_SRC}/${TAR_PYOPENCL} ]
    then
	echo "Downloading pyopencl source:..."
	if((ISTHERE_WGET))
	then
	    wget -P ${VENV_SRC} ${WGET_PYOPENCL} --no-check-certificate
	elif((ISTHERE_CURL))
	then	    
	    mkdir ${VENV_SRC}
	    curl -o ${VENV_SRC}/${TAR_PYOPENCL} ${WGET_PYOPENCL}	    
	fi
	echo "Done (Downloading pyopencl source)"
    fi
    
    echo "Configuring pyopencl"
    STARTDIR=${PWD}
    cd ${VENV_SRC}
    tar zxfk ${TAR_PYOPENCL}
    cd ${DIR_PYOPENCL}
    ${ID_PYTHON} configure.py

    ## If CUDA is found I need to modify the files
    ## Thanks to: https://wiki.tiker.net/PyOpenCL/Installation/Windows/
    
    if ((CUDA_F))
    then
	CUDA_PATH_SED=$(echo ${CUDA_PATH} | sed 's/\//\\\//g')
	SED_STR_1_OLD="CL\_INC\_DIR\ =\ \[\]"
	SED_STR_1_NEW="CL\_INC\_DIR\ =\ \[\'${CUDA_PATH_SED}\/include\'\]"
	SED_STR_2_OLD="CL\_LIB\_DIR\ =\ \[\]"
	SED_STR_2_NEW="CL\_LIB\_DIR\ =\ \[\'${CUDA_PATH_SED}\/lib64\'\]"
	echo ${SED_STR_1_NEW}
	echo ${SED_STR_2_NEW}

	sed -i.bkp "s/${SED_STR_1_OLD}/${SED_STR_1_NEW}/g" siteconf.py
	sed -i.bkp "s/${SED_STR_2_OLD}/${SED_STR_2_NEW}/g" siteconf.py
	cat siteconf.py
    fi
    
    echo "Building pyopencl"
    ${ID_PYTHON} setup.py build
    
    echo "Install pyopencl"
    ${ID_PYTHON} setup.py install
    
    cd ${STARTDIR}
fi
echo
echo "Running idpy/test.py to check the installation"

TEST_RES=$(python -m unittest -v idpy/test.py 1>${IDPY_TEST_STDOUT} 2>${IDPY_TEST_STDERR} && echo 1 || echo 0)
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
for((ALIAS_I=0; ALIAS_I<${#IDPY_ALIASES[@]}; ALIAS_I++))
do
	echo "${IDPY_ALIASES[ALIAS_I]}" >> ${VENV_ALIASES}
done
## Appending lines at the end of the .bashrc to source the aliases
echo "${ID_BASHRC_BANNER}" >> ${HOME}/.bashrc
echo "${ID_BASHRC_ALIASES_OPT}" >> ${HOME}/.bashrc
echo "${ID_BASHRC_SOURCE_ALIASES}" >> ${HOME}/.bashrc

echo "For using the alias in the present shell: source ${HOME}/.bashrc"

echo "Python virtual environment for idea.deploy initialized!"
echo
