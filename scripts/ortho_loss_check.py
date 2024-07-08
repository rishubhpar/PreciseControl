import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

import pickle
embeddings_path='./weights/clip_face_basis100k_pca_wo_mean.pkl'

with open(embeddings_path, 'rb') as f:
    clip_face_basis = pickle.load(f)


face_token_embeddings1 = clip_face_basis['clip_token1']['components']

face_token_embeddings2 = clip_face_basis['clip_token2']['components']

print("checking orthogonality of face embeddings: ", (face_token_embeddings1.T @ face_token_embeddings2).shape)