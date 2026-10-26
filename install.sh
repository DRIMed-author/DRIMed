#!/bin/bash
# Environment setup for DRIMed
set -e

conda create -n DRIMed python=3.9 -y
conda activate DRIMed
pip install -r requirements.txt

echo "DRIMed environment ready. Run: python main_DRIMed.py --model DRIMed"
