#!/usr/bin/env bash
# Clone (if needed), patch, build, and editable-install pymetallic for idpy Metal.
# Soft-fail: prints warnings and returns non-zero without aborting the caller.
# Copyright (C) 2020-2026 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f .idpy-env ]
then
    echo "WARNING: .idpy-env not found in ${REPO_ROOT}; cannot install pymetallic."
    exit 1
fi
# shellcheck source=/dev/null
source .idpy-env

PATCH_FILE="${REPO_ROOT}/patches/pymetallic/0001-recommended-max-working-set-size.patch"
PIP_SERVER_OPTION="${PIP_SERVER_OPTION:-}"

soft_fail() {
    echo "WARNING: $*"
    echo "WARNING: Metal backend will be unavailable."
    exit 1
}

if [[ "${IDEP_OS}" != "MacOS" ]]
then
    echo "Skipping pymetallic install (Metal is macOS-only; OS=${IDEP_OS})"
    exit 0
fi

if ! command -v swiftc >/dev/null 2>&1
then
    soft_fail "swiftc not found (install Xcode / Swift toolchain for Metal)"
fi

if [ ! -f "${PATCH_FILE}" ]
then
    soft_fail "missing patch file: ${PATCH_FILE}"
fi

mkdir -p "${SOURCES_ROOT}"

if [ ! -d "${PYMETALLIC_SRC}" ]
then
    echo "Cloning pymetallic from ${PYMETALLIC_REPO}..."
    if ! git clone "${PYMETALLIC_REPO}" "${PYMETALLIC_SRC}"
    then
        soft_fail "git clone failed for ${PYMETALLIC_REPO}"
    fi
    if ! git -C "${PYMETALLIC_SRC}" checkout "${PYMETALLIC_PIN}"
    then
        soft_fail "git checkout ${PYMETALLIC_PIN} failed"
    fi
elif [ -d "${PYMETALLIC_SRC}/.git" ]
then
    CURRENT_SHA="$(git -C "${PYMETALLIC_SRC}" rev-parse HEAD 2>/dev/null || echo "")"
    if [[ "${CURRENT_SHA}" != "${PYMETALLIC_PIN}" ]]
    then
        echo "WARNING: ${PYMETALLIC_SRC} is at ${CURRENT_SHA}, expected ${PYMETALLIC_PIN}"
        echo "WARNING: attempting checkout of pin..."
        git -C "${PYMETALLIC_SRC}" fetch origin "${PYMETALLIC_PIN}" 2>/dev/null || true
        if ! git -C "${PYMETALLIC_SRC}" checkout "${PYMETALLIC_PIN}"
        then
            echo "WARNING: could not checkout pin; continuing with existing tree"
        fi
    fi
else
    echo "WARNING: ${PYMETALLIC_SRC} exists but is not a git repo; attempting patch+build anyway"
fi

if [ ! -f "${PYMETALLIC_SRC}/src/SwiftMetalBridge.swift" ]
then
    soft_fail "pymetallic sources incomplete at ${PYMETALLIC_SRC}"
fi

echo "Applying idpy pymetallic patch (if needed)..."
if git -C "${PYMETALLIC_SRC}" apply --check "${PATCH_FILE}" >/dev/null 2>&1
then
    if ! git -C "${PYMETALLIC_SRC}" apply "${PATCH_FILE}"
    then
        soft_fail "failed to apply ${PATCH_FILE}"
    fi
    echo "Patch applied."
elif git -C "${PYMETALLIC_SRC}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1
then
    echo "Patch already applied; skipping."
else
    soft_fail "patch neither applies nor appears already applied: ${PATCH_FILE}"
fi

echo "Building local pymetallic Swift bridge (libpymetallic.dylib)..."
if ! (cd "${PYMETALLIC_SRC}" && make build)
then
    soft_fail "pymetallic make build failed"
fi

if ! command -v pip >/dev/null 2>&1
then
    soft_fail "pip not found (activate the idpy virtualenv first)"
fi

echo "Installing pymetallic editable from ${PYMETALLIC_SRC}..."
PIP_CMD=(pip)
if [ -n "${VENV:-}" ] && [ -x "${VENV}/bin/pip" ]
then
    PIP_CMD=("${VENV}/bin/pip")
fi
# Local editable install: drop SOCKS proxies so pip can fetch build deps without PySocks
if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    "${PIP_CMD[@]}" install -e "${PYMETALLIC_SRC}" ${PIP_SERVER_OPTION}
then
    soft_fail "pip install -e failed for pymetallic"
fi

echo "pymetallic (Metal) ready."
exit 0
