import os
import shutil
import torch
import numpy as np

from .utils import *


def copy_files_recursively(source_folder, destination_folder, new_filename=False):
    # Ensure that the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Create new filename starting from 0
    if new_filename:
        i = 0

    # Iterate through all files and subfolders in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)

            if new_filename:
                file = str(i) + '.' + file.split('.')[1]
                i += 1
            destination_path = os.path.join(destination_folder, file)

            # Copy the file to the destination folder
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {source_path} to {destination_path}")