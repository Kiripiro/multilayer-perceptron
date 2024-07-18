#!/bin/bash

VENV_DIR="venv"

if [ ! -f requirements.txt ]; then
  echo "Error: requirements.txt does not exist in the current directory."
  exit 1
fi

create_venv() {
  echo "Creating virtual environment..."
  python3 -m venv $VENV_DIR

  echo "Activating virtual environment..."
  source $VENV_DIR/bin/activate

  echo "Upgrading pip..."
  pip install --upgrade pip

  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt

  deactivate

  echo "The virtual environment has been successfully set up."
  echo "To activate the virtual environment, run: source $VENV_DIR/bin/activate"
}

if [ -d "$VENV_DIR" ]; then
  read -p "Virtual environment already exists. Do you want to remake it? (yes/no): " choice
  case "$choice" in 
    yes|Yes|y|Y ) 
      echo "Removing existing virtual environment..."
      rm -rf $VENV_DIR
      create_venv
      ;;
    no|No|n|N ) 
      echo "Keeping the existing virtual environment."
      echo "To activate the virtual environment, run: source $VENV_DIR/bin/activate"
      ;;
    * ) 
      echo "Invalid choice. Exiting."
      exit 1
      ;;
  esac
else
  create_venv
fi
