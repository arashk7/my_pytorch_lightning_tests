#!/bin/bash
set -e

# Update system and install basics
sudo apt-get update -y
sudo apt-get install -y git wget curl build-essential

# Clone your repo (if not already cloned by RunPod)
if [ ! -d "myproject" ]; then
    git clone https://github.com/arashk7/my_pytorch_lightning_tests.git myproject
fi
cd myproject

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
