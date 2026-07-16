# Script for cleaning idea.deploy python virtual environment
# Copyright (C) 2020-2022 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 19/2/2022

# For the time being, the verbose flag needs to be changed here
VERBOSE_FLAG=0

PRESERVE_SOURCES=0
PURGE_SOURCES=0

usage() {
    echo "Usage: bash idpy-clean.sh [--preserve-sources|--purge-sources] [-h|--help]"
    echo
    echo "  (default)             Remove the venv and run 'make clean' in pymetallic if present"
    echo "  --preserve-sources    Remove the venv only; keep sources and Metal build artifacts"
    echo "  --purge-sources       Remove the venv and delete py-env/sources entirely"
    echo "  -h, --help            Show this help"
}

while [ ${#} -gt 0 ]
do
    case "${1}" in
        --preserve-sources)
            PRESERVE_SOURCES=1
            shift
            ;;
        --purge-sources)
            PURGE_SOURCES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: ${1}"
            usage
            exit 1
            ;;
    esac
done

if ((PRESERVE_SOURCES)) && ((PURGE_SOURCES))
then
    echo "Error: --preserve-sources and --purge-sources are mutually exclusive"
    usage
    exit 1
fi

source .idpy-env
echo
echo "Welcome to the idea.deploy cleaning script!"
echo "Cleaning jupyter environment..."
(
    source "${VENV}/bin/activate"
    jupyter kernelspec uninstall idpy-env -y
)
echo "done"
echo

#################################
## Cleaning virtual environment
echo -n "Cleaning python virtual environment..."
if [ -d "${VENV}" ]
then
	if ((${VERBOSE_FLAG}))
	then
		echo
	    echo rm -r "${VENV}"
	fi		
	rm -r "${VENV}"
fi
echo "done"
echo
#################################
## Cleaning found cuda path
echo -n "Cleaning found cuda path..."
if [ -f "${ID_CUDA_PATH_FOUND}" ]
then
	if ((${VERBOSE_FLAG}))
	then
		echo
	    echo rm "${ID_CUDA_PATH_FOUND}"
	fi	
	rm "${ID_CUDA_PATH_FOUND}"
fi
echo "done"
echo
#################################
## Cleaning py-env/sources
if ((PURGE_SOURCES))
then
	echo -n "Purging sources under ${SOURCES_ROOT}..."
	if [ -d "${SOURCES_ROOT}" ]
	then
		if ((${VERBOSE_FLAG}))
		then
			echo
			echo rm -rf "${SOURCES_ROOT}"
		fi
		rm -rf "${SOURCES_ROOT}"
	fi
	echo "done"
	echo
elif ((PRESERVE_SOURCES))
then
	echo "Preserving sources and Metal build artifacts under ${SOURCES_ROOT}"
	echo
else
	echo -n "Cleaning Metal build artifacts..."
	if [ -d "${PYMETALLIC_SRC}" ] && [ -f "${PYMETALLIC_SRC}/Makefile" ]
	then
		if ((${VERBOSE_FLAG}))
		then
			echo
			echo "(cd \"${PYMETALLIC_SRC}\" && make clean)"
		fi
		(cd "${PYMETALLIC_SRC}" && make clean)
	fi
	echo "done"
	echo
fi
#################################
## Deleting python tests output
echo -n "Checking python tests outputs..."
FILES_LIST[0]="${IDPY_TEST_STDOUT}"
FILES_LIST[1]="${IDPY_TEST_STDERR}"
for((FILE_I=0; FILE_I<${#FILES_LIST[@]}; FILE_I++))
do
	FILE_NAME=${FILES_LIST[${FILE_I}]}
    if [ -f "${FILE_NAME}" ]
    then
    	if ((${VERBOSE_FLAG}))
    	then
    		echo
		    echo rm "${FILE_NAME}"
		 fi
	    rm "${FILE_NAME}"
	fi
done
echo "done"
echo
#################################
## Cleaning .bashrc
echo -n "Checking for aliases sourcing in ${HOME}/.bashrc ..."
STRING_LIST[0]="${ID_BASHRC_BANNER}"
STRING_LIST[1]=${ID_BASHRC_ALIASES_OPT} 
STRING_LIST[2]=${ID_BASHRC_SOURCE_ALIASES}
for((STRING_I=0; STRING_I<${#STRING_LIST[@]}; STRING_I++))
do
	STRING=${STRING_LIST[${STRING_I}]}
	STRING_SED=$(echo ${STRING} | sed 's/\//\\\//g' | sed 's/\ /\\ /g')
	sed -i.idpy.bkp -e "/${STRING_SED}/d" ${HOME}/.bashrc
done
echo "done"
echo
echo "idea.deploy python virtual environment has been successfully cleaned!"
