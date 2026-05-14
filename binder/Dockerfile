# syntax=docker/dockerfile:1.7
FROM python:3.13-slim-bookworm

# Binder requires a user named jovyan with UID 1000.
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER}

RUN groupadd -g ${NB_UID} ${NB_USER} \
 && useradd --create-home --uid ${NB_UID} --gid ${NB_UID} --shell /bin/bash ${NB_USER}

# System libraries required by napari/Qt, h5py, and OpenGL software rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglu1-mesa \
        libx11-xcb1 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Pin uv to the base image's Python so the repo's .python-version (3.14) does
# not trigger a managed-Python download of a version with no cartopy wheels.
ENV UV_PYTHON=3.13 \
    UV_PYTHON_DOWNLOADS=never

COPY --chown=${NB_UID}:${NB_UID} . ${HOME}/
USER ${NB_USER}
WORKDIR ${HOME}

# --no-dev keeps pytest/ruff/ty out of the image; notebook (>=7 bundles
# JupyterLab) provides the `jupyter notebook` CLI that repo2docker launches.
RUN uv sync --frozen --no-dev \
 && uv pip install notebook jupytext ipykernel ipywidgets

# Server-side: register jupytext as the contents manager so percent-format .py
# files are exposed as notebooks over the API.
RUN mkdir -p ${HOME}/.jupyter \
 && printf '%s\n' \
        'c.ServerApp.contents_manager_class = "jupytext.TextFileContentsManager"' \
        'c.ContentsManager.preferred_jupytext_formats_read = "py:percent"' \
        > ${HOME}/.jupyter/jupyter_server_config.py

# Frontend: make JupyterLab open .py files with the Jupytext Notebook viewer
# by default, so ?urlpath=lab/tree/<file>.py lands in a notebook view.
RUN mkdir -p ${HOME}/.jupyter/labconfig \
 && printf '%s\n' \
        '{' \
        '  "@jupyterlab/docmanager-extension:plugin": {' \
        '    "defaultViewers": {' \
        '      "python": "Jupytext Notebook"' \
        '    }' \
        '  }' \
        '}' \
        > ${HOME}/.jupyter/labconfig/default_setting_overrides.json

ENV PATH=${HOME}/.venv/bin:${PATH}
