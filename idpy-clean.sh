# Script for cleaning idea.deploy python virtual environment
# Copyright (C) 2020 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 28/8/2020

source .idpy-env
echo
echo "Welcome to the idea.deploy cleaning script!"
echo "Cleaning python virtual environment..."
echo "Launching subshell for cleaning jupyter environment"
(
    source ${VENV}/bin/activate
    jupyter kernelspec uninstall idpy-env
)

START_DIR=${PWD}
# We change directory
echo
echo "Entering ${VENV}"
echo 
cd ${VENV}

# Creting an array with the files/directories that should not be deleted
# i.e. those not-ignored (!) by the .gitignore file in the ${VENV} directory
SAFE_LIST=$(grep "^\!" .gitignore | cut -d '!' -f 2)
echo "List of files and directories that will NOT be deleted"
SAFE_LIST=$(echo "${SAFE_LIST}")
echo ${SAFE_LIST}
echo

if [ ${IDEP_OS} == "Linux" ]
then
    FIND_CUT_F=2
elif [ ${IDEP_OS} == "MacOS" ]
then
    FIND_CUT_F=3
fi
LOCAL_DIRS=$(find ./ -maxdepth 1 -type d | cut -d '/' -f ${FIND_CUT_F} | tail -n +2)
LOCAL_FILES=$(find ./ -maxdepth 1 -type f | cut -d '/' -f ${FIND_CUT_F} | tail -n +2)

########################
### DELETING DIRECTORIES
echo "Removing other directories..."
echo
COUNT=0
for N in ${LOCAL_DIRS}
do
    SAFE_F=0
    for M in ${SAFE_LIST}
    do
	if [ ${M} == ${N} ]
	then
	    SAFE_F=1
	fi
    done
    
    if ((${SAFE_F} == 0))
    then
	echo "${N}"
	((COUNT++))
    fi
done

if((COUNT))
then
    echo
    RM_REPLY=0
    while true
    do
	read -p "Does this look fine? (Y/N) " yn
	case ${yn} in
	    [Yy]* ) RM_REPLY=1; break;;
	    [Nn]* ) break;;
	    * ) echo "Please answer yes or no";;
	esac
    done
    echo
    if((RM_REPLY))
    then
	for N in ${LOCAL_DIRS}
	do
            SAFE_F=0
            for M in ${SAFE_LIST}
            do
		if [ ${M} == ${N} ]
		then
		    SAFE_F=1
		fi
            done
	    
            if ((${SAFE_F} == 0))
            then
		echo "rm -rf ${N}"
		rm -rf "${N}"
            fi
	done
    fi
    echo
    echo "...done"
else
    echo "...nothing to delete"
fi
echo
echo
########################
### DELETING FILES
echo "Removing other files..."
echo
COUNT=0
for N in ${LOCAL_FILES}
do
    SAFE_F=0
    for M in ${SAFE_LIST}
    do
	if [ ${M} == ${N} ]
	then
	    SAFE_F=1
	fi
    done
    
    if ((${SAFE_F} == 0))
    then
	echo "${N}"
	((COUNT++))
    fi
done

if((COUNT))
then
    echo
    RM_REPLY=0
    while true
    do
	read -p "Does this look fine? (Y/N) " yn
	case ${yn} in
	    [Yy]* ) RM_REPLY=1; break;;
	    [Nn]* ) break;;
	    * ) echo "Please answer yes or no";;
	esac
    done
    echo
    if((RM_REPLY))
    then
	for N in ${LOCAL_FILES}
	do
            SAFE_F=0
            for M in ${SAFE_LIST}
            do
		if [ ${M} == ${N} ]
		then
		    SAFE_F=1
		fi
            done
	    
            if ((${SAFE_F} == 0))
            then
		echo "rm ${N}"
		rm "${N}"
            fi
	done
    fi
    echo
    echo "...done"
else
    echo "...nothing to delete"
fi
echo
echo "Going back to initial directory"
echo "${START_DIR}"
echo
#################################
## Deleting python tests output
echo "Checking python tests outputs..."
FILES_LIST="${IDPY_TEST_STDOUT} ${IDPY_TEST_STDERR}"
for FILE_NAME in ${FILES_LIST}
do
    echo "${FILE_NAME}"
    if [ -f ${FILE_NAME} ]
    then
	echo "Remove ${FILE_NAME}"
	RM_REPLY=0
	while true
	do
	    read -p "Does this look fine? (Y/N) " yn
	    case ${yn} in
		[Yy]* ) RM_REPLY=1; break;;
		[Nn]* ) break;;
		* ) echo "Please answer yes or no";;
	    esac
	done
	if((RM_REPLY))
	then
	    echo "rm ${FILE_NAME}"
	    rm "${FILE_NAME}"
	fi
	echo
    fi
done

echo "Checking for aliases in ${HOME}/.bashrc"
## ALIASES
for((ALIAS_I=0; ALIAS_I<${#IDPY_ALIASES[@]}; ALIAS_I++))
do
    ALIAS_STRING=${IDPY_ALIASES[ALIAS_I]}
    ALIAS_STRING_SED=${IDPY_ALIASES_SED[ALIAS_I]}
    ALIAS_CHECK=$(grep "${ALIAS_STRING_SED}" ${HOME}/.bashrc \
			1>/dev/null 2>/dev/null && echo 1 || echo 0)
    if((${ALIAS_CHECK} == 1))
    then
	echo "Deleting aliases..."
	echo "${ALIAS_STRING}"
	sed -i.idpy.bkp -e "/${ALIAS_STRING_SED}/d" ${HOME}/.bashrc
    fi
done
# We get back from where we started
cd ${START_DIR}
