import os
import cv2 
import numpy as np


dataset_path = "../CelebBasisV2/aug_images/pose/"

file_list = os.listdir(dataset_path)

for file in file_list:
    img = cv2.imread(os.path.join(dataset_path, file))
    flipped_img = cv2.flip(img, 1)
    os.rename(os.path.join(dataset_path, file), os.path.join(dataset_path, file.split(".")[0]+"_0.png"))
    cv2.imwrite(os.path.join(dataset_path, file.split(".")[0]+"_1.png"), flipped_img)

print("Done")