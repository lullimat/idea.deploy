# Script for bootstrapping the installation of idea.deploy python virtual environment
# Copyright (C) 2020-2025 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 10/6/2025

echo "Bootstrapping idea.deploy"
echo

SCRIPT_DIR="${PWD}"
IDPY_REPO="https://github.com/lullimat/idea.deploy.git"
INSTALL_DIR="${HOME}"
echo "Cloning idea.deploy into default directory" "${INSTALL_DIR}"
echo
while true
do
    read -p "If you wish to change type the directory path (press 'return' for default): " INSTALL_DIR
    if [ -z ${INSTALL_DIR} ]
    then
        INSTALL_DIR="${HOME}"
    fi
    # Check if path exists
    IS_INSTALL_DIR=$(test -d "${HOME}" && echo 1 || echo 0)
    if((IS_INSTALL_DIR == 0))
    then
        echo "Install directory does not exist, exiting"
        exit
    else
        break
    fi
done

IDPY_DIR="${INSTALL_DIR}/idea.deploy"

GIT_F=$(command -v git > /dev/null 2>&1 && echo 1 || echo 0)
if((GIT_F == 0))
then
    echo "git is not installed!"
    exit
fi

git clone "${IDPY_REPO}" "${IDPY_DIR}"
cd "${IDPY_DIR}"

bash idpy-init.sh

cd "${SCRIPT_DIR}"