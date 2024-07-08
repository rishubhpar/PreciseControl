from PIL import Image
import os

# folder_path = '/media/tets/rishubh/sachi/CelebBasis_pstar/aug_images/attributes/'
# save_path = '/media/tets/rishubh/sachi/CelebBasis_pstar/aug_images/attributes_new/'

# if(not os.path.exists(save_path)):
#     os.mkdir(save_path)

# for folder in os.listdir(folder_path):
#     for file in os.listdir(os.path.join(folder_path, folder)):
#         img = Image.open(os.path.join(folder_path, folder, file))
#         img = img.convert('RGB')
#         if(not os.path.exists(os.path.join(save_path, folder))):
#             os.mkdir(os.path.join(save_path, folder))
#         img.save(os.path.join(save_path, folder, file))


img_folder_path = '/media/tets/rishubh/sachi/CelebBasis_pstar_sd2/cross_attn_at_each_timestep/'
save_path = '/media/tets/rishubh/sachi/CelebBasis_pstar_sd2/cross_attn_videos'
os.makedirs(save_path, exist_ok=True)

import cv2 
import numpy as np


imgs_path = sorted(os.listdir(img_folder_path))
video_list1 = sorted(imgs_path[:len(imgs_path)//2])
video_list2 = sorted(imgs_path[len(imgs_path)//2:])
print(video_list1)
img_array = []
img_array2 = []
for filename in video_list1:
    img = cv2.imread(os.path.join(img_folder_path,filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img.astype(np.uint8)[:,:,::-1])

for filename in video_list2:
    img = cv2.imread(os.path.join(img_folder_path,filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array2.append(img.astype(np.uint8)[:,:,::-1])

# out = cv2.VideoWriter(os.path.join(save_path,'./cross_attn_1.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
# out2 = cv2.VideoWriter(os.path.join(save_path, './cross_attn_2.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

# for i in range(len(img_array2)):
#     out2.write(img_array2[i])
# out2.release()
import imageio
imageio.mimsave(os.path.join(save_path, 'cross_attn_astar_1.gif'), img_array, duration=0.2)
imageio.mimsave(os.path.join(save_path, 'cross_attn__atar_2.gif'), img_array2, duration=0.2)

print('Done')