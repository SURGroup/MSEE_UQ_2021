import fire
import os
from os.path import dirname
import os.path
from os import path

def run_model(index):
    
    # Step 1: Copy the generated input file in the parent directory i.e., where the script simple.py exists.
    # Step 2: Run the model.
    # Step 3: Create OutputFiles folder in the current directory.
    # Step 4: Copy the generated *.vtk file in the OutputFiles folder.
    
    pass



if __name__=='__main__':
    fire.Fire(run_model)


