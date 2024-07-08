"""
This script will combine the results from all the methods and store in a single grid of images 
"""
import os
import cv2 
import numpy as np 
import random


def shuffle_img(img, n_imgs): 
    h_raw, w_raw = img.shape[0], img.shape[1] 
    h_img, w_img = int(2 * h_raw / 3), int(w_raw / n_imgs)

    edit_imgs, edit_imgs_label = [], []
    for id in range(1, n_imgs):
        start_w = w_img*id
        end_w = w_img*(id+1) 

        edit_img = img[int(h_raw/3):,start_w:end_w,:]
        edit_imgs.append(edit_img)

        edit_img_w_label = img[:,start_w:end_w,:]
        edit_imgs_label.append(edit_img_w_label)

    # Shuffling the edits and accumulating them in a single stack 
    combined_list = list(zip(edit_imgs, edit_imgs_label))
    random.shuffle(combined_list) 
    edit_imgs, edit_imgs_labeled = zip(*combined_list)
    edit_imgs, edit_imgs_labeled = list(edit_imgs), list(edit_imgs_labeled) 

    # Adding the first image in the stack 
    input_img = img[int(h_raw/3):, :w_img, :] # Taking the first image out and not shuffling it
    
    # final list of edit stacks shuffled with and without labels 
    edit_imgs = [input_img] + edit_imgs
    edit_imgs_labeled = [img[:,:w_img,:]] + edit_imgs_labeled 

    edit_imgs = np.hstack(edit_imgs)
    # We have to add caption to the images for processing 
    imgs_buffer = np.zeros((int(edit_imgs.shape[0]/4), edit_imgs.shape[1], edit_imgs.shape[2]))
    # print("img buffer shape: {}".format(imgs_buffer.shape))
    imgs_buffer = cv2.putText(imgs_buffer, '     Input            a)              b)             c)               d)               e)                f) ', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 9, cv2.LINE_AA)
    edit_imgs = np.vstack([imgs_buffer, edit_imgs])

    edit_imgs_labeled = np.hstack(edit_imgs_labeled) 
    return edit_imgs, edit_imgs_labeled

# This is the main function that will load the images and create a grid of output images from all the methods 
def run_main(): 
    # root_fld = './benchmark_results'
    root_fld = './benchmark_results/user_study_imgs/spliced_images'
    dst_path = './benchmark_results/user_study_imgs/user_study_shuffled'
    dst_path_labeled = './benchmark_results/user_study_imgs/user_study_shuffled_labeled' 

    idx = 0
    # Iterating over the images and selecting some images randomly based on fixed probability 
    for img_nm in os.listdir(root_fld):
        rand_num = np.random.randint(0,100)
        if (rand_num > 92):
            idx+=1 
            img_pt = os.path.join(root_fld, img_nm)
            img = cv2.imread(img_pt)

            edit_imgs, edit_imgs_labeled = shuffle_img(img, n_imgs=7)
            edit_imgs_pt = os.path.join(dst_path, img_nm) 
            edit_imgs_labeled_pt = os.path.join(dst_path_labeled, img_nm) 

            # Saving the images in the defined paths 
            cv2.imwrite(edit_imgs_pt, edit_imgs)
            cv2.imwrite(edit_imgs_labeled_pt, edit_imgs_labeled) 
    
    print("number of imgs sampled - :{}".format(idx)) 

if __name__ == "__main__":
    run_main() 