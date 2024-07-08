import os
import numpy as np
import torch
import shutil

import sys
sys.path.append("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/evaluation/clean-fid")
from cleanfid import fid
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


gen_img_dataset_path = "./benchmark_ravdess_results/w_adapter"
dataset_name = gen_img_dataset_path.split("/")[-1]
src_img_dataset_path = "../../datasets/coco2017_person_class"
custom_stats_available = True

# precompute coco dataset metrics, run once
# fid.make_custom_stats("coco_person_fid", src_img_dataset_path, mode="legacy_pytorch")


used_prompt_list = ["Manga drawing of sks person", "Ukiyo-e painting of sks person", "Cubism painting of sks person", 
               "Banksy art of sks person", "Cave mural depicting of sks person", 
               "sand sculpture of sks person", "Colorful graffiti of a sks person", 
               "Watercolor painting of sks person", "Pointillism painting of sks person", "sks person stained window glass", 
               "sks person latte art", "sks person as a greek sculpture", 
               "sks person as a knight in plate armour", "sks person wears a scifi suit in space", 
               "sks person in a chinese vegetable market", "sks person in a superhero costume", "sks person on the beach", 
               "sks person is playing the guitar", "sks person eats bread in front of the Eiffel Tower", 
               "sks person wearing yellow jacket, and driving a motorbike", "sks person wearing a santa hat in a snowy forest",
               "sks person wearing a black robe with a light saber in hand", "sks person as a firefighter in front of a burning building",
               "sks person camping in the woods in front of a campfire", "sks person taking a selfie in front of grand canyon"]


# create folder to copy a gen_imgs
cache_gen_folder = "./fid_cache_folder"
shutil.rmtree(cache_gen_folder, ignore_errors=True)
os.makedirs(cache_gen_folder, exist_ok=True)

id_list = sorted(os.listdir(gen_img_dataset_path))
prompt_list = os.listdir(os.path.join(gen_img_dataset_path, id_list[0]))
num_of_samples = len(os.listdir(os.path.join(gen_img_dataset_path, id_list[0], prompt_list[0])))
for id_name in id_list:
    for prompt in prompt_list:
        if prompt not in used_prompt_list:
            print(f"Skipping {prompt}")
            continue
        img_id = np.random.randint(0, num_of_samples)
        img_file_name = os.listdir(os.path.join(gen_img_dataset_path, id_name, prompt))[img_id]
        shutil.copyfile(os.path.join(gen_img_dataset_path, id_name, prompt, img_file_name), os.path.join(cache_gen_folder, f"{id_name}_{prompt}_{img_file_name}"))


if(custom_stats_available):
    score = fid.compute_fid(cache_gen_folder, dataset_name="coco_person_fid", mode="legacy_pytorch",  dataset_split="custom")
    print(f"FID score of {dataset_name}: ", score)
    kid_score = fid.compute_kid(cache_gen_folder, dataset_name="coco_person_fid", mode="legacy_pytorch",  dataset_split="custom")
    print(f"KID score of {dataset_name}: ", kid_score)

else:
    score = fid.compute_fid(cache_gen_folder, src_img_dataset_path, mode="legacy_pytorch")
    print(f"FID score of {dataset_name}: ", score)

    kid_score = fid.compute_kid(cache_gen_folder, src_img_dataset_path, mode="legacy_pytorch")
    print(f"KID score of {dataset_name}: ", kid_score)