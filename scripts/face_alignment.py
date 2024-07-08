import dlib 
import os
import PIL.Image as Image

import sys
sys.path.append("./")
from ldm.modules.e4e.alignment import align_face
import shutil

shape_predictor = dlib.shape_predictor('./weights/encoder/shape_predictor_68_face_landmarks.dat')
# img_dir = "./attr_edit_eval_test/ours_text_samples"
# save_dir = "./attr_edit_eval_test/ours_text_samples/edited"
img_dir = "./aug_images/bindi_imgs"
save_dir = "./aug_images/bindi_imgs/edited"
shutil.rmtree(save_dir, ignore_errors=True)

img_list = os.listdir(img_dir)
print("Total images: ", img_list)
# img_list = [img_list[3]]
for img_name in img_list:
    print("Processing: ", img_name)
    img_path = os.path.join(img_dir, img_name)
    save_path = os.path.join(save_dir, img_name)
    aligned_img = align_face(img_path, shape_predictor)
    os.makedirs(save_dir, exist_ok=True)
    if(aligned_img is None):
        print("Failed to align: ", img_name)
        continue
    aligned_img = aligned_img.resize((512, 512), Image.BICUBIC)
    aligned_img.save(save_path)
    
    
