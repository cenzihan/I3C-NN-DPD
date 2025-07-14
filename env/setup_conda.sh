#!/bin/bash

# Script to create and set up the conda environment for the NN-DPD project.
# It uses the 'environment.yml' file located in the same directory as the script.

# Get the directory where the script is located. This makes the script runnable from anywhere.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Error: Conda is not installed or not in your PATH."
    echo "Please install Anaconda or Miniconda first: https://www.anaconda.com/products/distribution"
    exit 1
fi

ENV_NAME="nn-dpd-env"
# The yml file is now referenced relative to the script's location.
ENV_FILE="$SCRIPT_DIR/environment.yml"

echo "Attempting to create conda environment '$ENV_NAME' from '$ENV_FILE'..."

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME"
    else
        echo "Aborting. To update the existing environment, you can run:"
        echo "conda env update --name $ENV_NAME --file $ENV_FILE --prune"
        exit 0
    fi
fi

# Create the environment from the yml file
conda env create -f "$ENV_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "Conda environment '$ENV_NAME' created successfully."
    echo "To activate the environment, run:"
    echo "conda activate $ENV_NAME"
else
    echo ""
    echo "Error: Failed to create the conda environment. Please check for errors above."
    exit 1
fi 