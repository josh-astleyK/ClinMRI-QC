import argparse
import os
import sys
import numpy as np 
import nibabel

def load_nifti (path:str) -> np.ndarray: 
    img = nibabel.load(path)
    data = np.asanyarray(img.dataobj, dtype = np.float32)
    return data 
