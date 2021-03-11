# Script for initializing idea.deploy python virtual environment
# Copyright (C) 2020 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 28/8/2020

source .idpy-env
WGET_PYOPENCL_2020=https://files.pythonhosted.org/packages/a1/b5/c32aaa78e76fefcb294f4ad6aba7ec592d59b72356ca95bcc4abfb98af3e/pyopencl-2020.2.tar.gz
WGET_PYOPENCL_2021=https://files.pythonhosted.org/packages/71/2f/e5c0860f86f8ea8d8044db7b661fccb954c200308d94d982352592eb88ee/pyopencl-2021.1.2.tar.gz
WGET_PYOPENCL=${WGET_PYOPENCL_2021}

WGET_PYCUDA_2019=https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
WGET_PYCUDA_2020=https://files.pythonhosted.org/packages/46/61/47d3235a4c13eec5a5f03594ddb268f4858734e02980afbcd806e6242fa5/pycuda-2020.1.tar.gz
WGET_PYCUDA=${WGET_PYCUDA_2020}

TAR_PYOPENCL=$(echo ${WGET_PYOPENCL} | tr '/' ' ' | awk '{print($NF)}')
DIR_PYOPENCL=${TAR_PYOPENCL:0:${#TAR_PYOPENCL} - 7}
TAR_PYCUDA=$(echo ${WGET_PYCUDA} | tr '/' ' ' | awk '{print($NF)}')
DIR_PYCUDA=${TAR_PYCUDA:0:${#TAR_PYCUDA} - 7}


# Script scope
# Load local python env if None

# install requirements

# download pyopencl

# configure and build and install

# Never forget that ${VENV} is a global path

echo "Welcome to idea.deploy!"

## Check if Python3 is installed
echo -n "Checking python3 installation:... "
## Python3
PY3_F=$(command -v python3 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p5_F=$(command -v python3.5 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p6_F=$(command -v python3.6 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p7_F=$(command -v python3.7 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p8_F=$(command -v python3.8 >/dev/null 2>&1 && echo 1 || echo 0)
PY3p9_F=$(command -v python3.9 >/dev/null 2>&1 && echo 1 || echo 0)

##
if ((${PY3p8_F}))
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
if [ -f ${VENV}/cuda_paths ]
then
    CUDA_F=0
    while IFS= read -r line
    do
	if [ -d ${line} ]
	then
	    CUDA_F=1
	    CUDA_PATH=${line}
	fi
    done < ${VENV}/cuda_paths
else
    echo "File ${VENV}/cuda_paths not found!"
    echo "Looks like something is wrong with the installation"
    echo "Bye Bye!"
    exit 1
fi

if ((CUDA_F))
then
    echo "Found ${CUDA_PATH}"
    echo ${CUDA_PATH} > ${VENV}/cuda_path_found
    CUDA_EXPORT_PATH_STRING="export PATH=\${PATH}:${CUDA_PATH}/bin"
    CUDA_EXPORT_LD_LIB_PATH_STRING="export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${CUDA_PATH}/lib:${CUDA_PATH}/lib64"
    CHECK_CUDA_PATH_ACTIVATE=$(grep "${CUDA_EXPORT_PATH_STRING}" \
				    ${VENV_BIN}/activate \
				    1>/dev/null 2>/dev/null \
				   && echo 1 || echo 0)
    CHECK_CUDA_LD_PATH_ACTIVATE=$(grep "${CUDA_EXPORT_LD_LIB_PATH_STRING}" \
				       ${VENV_BIN}/activate \
				       1>/dev/null 2>/dev/null \
				      && echo 1 || echo 0)

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
    echo > ${VENV}/cuda_path_found
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
MPICC_F=$(which mpicc 1>/dev/null 2>/dev/null && echo 1 || echo 0)

echo "Sourcing virtual environment"
source ${VENV}/bin/activate

if((VENV_F == 0))
then
    echo "Pip installing requiremnts"
    pip install --upgrade pip setuptools wheel
    pip install -r ${VENV}/requirements.txt
    ## Install pycuda if cuda is found
    if ((CUDA_F && 0))
    then
	pip install pycuda
    fi
    ## Install mpi4py if mpicc is found
    if((MPICC_F))
    then
	pip install mpi4py
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
	wget -P ${VENV_SRC} ${WGET_PYCUDA}
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
	wget -P ${VENV_SRC} ${WGET_PYOPENCL}
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
    ALIAS_STRING=${IDPY_ALIASES[ALIAS_I]}
    ALIAS_CHECK=$(grep "${ALIAS_STRING}" ${HOME}/.bashrc \
			1>/dev/null 2>/dev/null && echo 1 || echo 0)

    if((${ALIAS_CHECK} == 0))
    then
	echo "${ALIAS_STRING}"
	ALIAS_REPLY=0
	while true
	do
	    read -p "Would you like to append an this alias to your ${HOME}/.bashrc? (Y/N) " yn
	    case ${yn} in
		[Yy]* ) ALIAS_REPLY=1; break;;
		[Nn]* ) exit;;
		* ) echo "Please answer yes or no";;
	    esac
	done
	if((${ALIAS_REPLY}))
	then
	    echo "${ALIAS_STRING}" >> ${HOME}/.bashrc
	fi
	echo "For using the alias in the present shell: source ${HOME}/.bashrc"
    fi
    echo
done
echo "Python virtual environment for idea.deploy initialized"
