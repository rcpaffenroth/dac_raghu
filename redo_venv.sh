#! /bin/bash

rm -rf venv
# This is know to work with
# https://repo.continuum.io/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
$HOME/minimamba/bin/python3 -m venv venv
# /usr/bin/python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt