import json
from pathlib import Path
import os
# import torch
from PIL import Image
import sys
sys.path.append("/home/test/anaconda3/envs/datid3d/lib/python3.9/site-packages/")

from stellar_metrics import metric_names, run_metric

root_generated_dir = "./benchmark_ravdess_results/celebbasis_sd21"
root_orig_images_dir = "./aug_images/ravdess_lora_comparision_data/"
num_of_samples = 1

id_list = sorted(os.listdir(root_orig_images_dir))

prompt_list = os.listdir(os.path.join(root_generated_dir, id_list[0]))
print("prompt_list: ", prompt_list)

for id_name in id_list:
    for prompt_name in prompt_list:
        print("id_name: ", id_name)
        print("prompt_name: ", prompt_name)
        
        generated_img_path = os.path.join(root_generated_dir, id_name, prompt_name)
        orig_img_path = os.path.join(root_orig_images_dir, id_name, "0000")
        print("generated_img_path: ", generated_img_path)
        print("orig_img_path: ", orig_img_path)

        syn_path = Path(generated_img_path)
        stellar_data_root = Path(orig_img_path)

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        df = run_metric(
            metric='ips',
            stellar_path=stellar_data_root,
            syn_path=syn_path,
            device=DEVICE,
            batch_size=1,
        )
        print("df: ", df)
        break
    break