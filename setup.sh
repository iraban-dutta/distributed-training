#!/bin/bash
# bash setup.sh

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

echo "Environment created. Activate with:"
echo "source .venv/bin/activate"