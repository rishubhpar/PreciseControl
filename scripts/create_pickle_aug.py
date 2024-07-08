import os
import cv2
import numpy as np
import pickle

data_dir = "./infer_images/dataset_stylegan3_10id/ffhq/"

save_dir = "./aug_images/dataset_stylegan3_10id/aug/"

if(not os.path.exists(save_dir)):
    os.makedirs(save_dir,exist_ok=True)

num_of_identity_needed = 3

num_of_identity = 0
pickle_list = []
for file_name in os.listdir(data_dir):
    if(num_of_identity == num_of_identity_needed):
        break
    img = cv2.imread(os.path.join(data_dir,file_name))
    img_flip = cv2.flip(img, 1)

    cv2.imwrite(os.path.join(save_dir,file_name), img_flip)
    pickle_list.append(os.path.join(save_dir,file_name))
    num_of_identity += 1

with open(os.path.join(save_dir,"ffhq.pkl"), "wb") as fp:
	pickle.dump(pickle_list, fp)
        
interpolated_images = []