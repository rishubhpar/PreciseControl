"""
This script will compose images from various methods for multi-person cases.
It will load images from Celeb-Basis, Custom-Diffusion, Textual-Inversion and Ours for qualitative comparison
and save them into a single stack 
"""

import os
import cv2 
import numpy as np 
import random