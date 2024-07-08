"""
This script will combine the results from all the methods and store in a single grid of images 
"""
import os
import cv2 
import numpy as np 
import random 

# This function load all the images for the given identity and all the methods and combine them into a single stack 
def combine_stack(root_fld, save_fld_path, save_fld_spliced_path, method_list, input_img, id_name):
    ignore_caption_list = ['sks person paragliding in the sky', 'sks person playing tennis at wimbledon',
                           'sks person kayaking through a small river, closeup', 'a sks person carrying plastic bags of vegetables in chinese vegetable market, sunny day',
                           'sks person doing freestyle dance in the streets', 'sks person as a knight in plate armour', 
                           'sks person as a wizard', 'sks person funko pop', 'sks person in a comic book', 
                           'sks person shaking hands with sks person', 'Ukiyo-e painting of sks person']
    caption_list = [txt for txt in os.listdir(os.path.join(root_fld, method_list[0], id_name)) if txt not in ignore_caption_list] 
    grid_out = []

    # Iterating over all the captions and combining them for all the methods in a single output images 
    for caption in caption_list:
        # print("ingput image shape: {}".format(input_img.shape))
        imgs_caption = [input_img]
        for method in method_list:
            img_fld_path = os.path.join(root_fld, method, id_name, caption)
            # print("img fld path: {}, exists: {}".format(img_fld_path, os.path.exists(img_fld_path)))

            # Loading the first image and then breaking to collate the outputs 
            for imgnm in sorted(os.listdir(img_fld_path), key=lambda x: random.random()):
                imgpt = os.path.join(img_fld_path, imgnm)
                img = cv2.imread(imgpt) 
                break

            if (img is None):
                img = np.zeros((512,512,3))
            
            # print("img shape: {}".format(img.shape))
            imgs_caption.append(img)

        imgs_caption = np.hstack(imgs_caption)  
        imgs_buffer = np.zeros((int(imgs_caption.shape[0]/2), imgs_caption.shape[1], imgs_caption.shape[2]))
        # print("img buffer shape: {}".format(imgs_buffer.shape))
        imgs_buffer = cv2.putText(imgs_buffer, '                              {}                        '.format(caption), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 9, cv2.LINE_AA)
        imgs_buffer = cv2.putText(imgs_buffer, '       Input          CD              DB           DB_lora            TI         CELEBBASIS      Ours-lora2 ', (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8, cv2.LINE_AA)

        # Saving each of the row seperately for qualitative results 
        split_img = np.vstack([imgs_buffer, imgs_caption])
        save_path = os.path.join(save_fld_spliced_path, id_name + '_' + caption + '.jpg')
        
        # print("save path: {}".format(save_path))
        cv2.imwrite(save_path, split_img)
        # exit()

        grid_out.append(imgs_buffer)         
        grid_out.append(imgs_caption)

    grid_out = np.vstack(grid_out)
    save_path = os.path.join(save_fld_path, id_name + '.jpg')
    # print("Saved image at: {}".format(save_path))
    # cv2.imwrite(save_path, grid_out)

# This is the main function that will load the images and create a grid of output images from all the methods 
def run_main(): 
    # root_fld = './benchmark_results'
    root_fld = './benchmark_results/user_study_imgs/src_data/5_imgs'
    dst_path = './benchmark_results/user_study_imgs/combined_imgs'
    dst_path_separate = './benchmark_results/user_study_imgs/spliced_images'
    src_img_path = './benchmark_results/user_study_imgs/input_imgs/edited'
    method_list = ['custom_diffusion5_10', 'db', 'db_lora', 'ti', 'celebbasis', 'ours_lora2'] 

    id_names = [fn for fn in os.listdir(os.path.join(root_fld, 'ti'))]
    input_imgs = [cv2.imread(os.path.join(src_img_path, id_nm) + '.jpg') for id_nm in id_names] 


    # for id in range(0, len(id_names)):
    #     print("id name: {}".format(id_names[id]))
    #     print("img shape: {}".format(input_imgs[id].shape))

    # exit()

    input_imgs = [cv2.resize(ipim, (512,512)) for ipim in input_imgs]

    # print("id nmaes: {}".format(id_names))

    for id in range(0, len(id_names)):
        combine_stack(root_fld, dst_path, dst_path_separate, method_list, input_imgs[id], id_names[id])

# This function will add gaussian noise on the source image for the visualziation purpose
def generate_noise_vis():
    img_pt = './debug_res_vis/two-person.jpg'
    img = cv2.imread(img_pt)
    print("img shape: {}".format(img.shape))

    gauss_img = np.random.randn(img.shape[0],img.shape[1],img.shape[2]) * 255.0 
    alpha = 0.7
    blend_img = alpha * img + (1-alpha) * gauss_img

    save_img_pt = './debug_res_vis/two-person-noise-0.7.jpg'
    cv2.imwrite(save_img_pt, blend_img) 

if __name__ == "__main__":
    # run_main()
    generate_noise_vis() 