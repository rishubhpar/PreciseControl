import os
import subprocess
import argparse

all_id_path = "./aug_images/comparision/edited/"

all_id = sorted(os.listdir(all_id_path))
all_id = ["einstein.jpg"]
# all_id = all_id[len(all_id)//2:]
# all_id = ["gates.jpg", "ed.jpg","lecun.jpg","bengio.jpg","fei.jpg","hinton.jpg"]
# all_id = ["hinton.jpg", "lecun.jpg","altman.jpg","musk.jpg","einstein.jpg","taylor.jpg"]

# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES=0 bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" "./infer_images/comparison_prompt.txt" "cnr" "0 0 0 0" True 4 49 0.3


argparser = argparse.ArgumentParser()
argparser.add_argument("--weight_id", type=int, default=49)
argparser.add_argument("--lora_scale", type=float, default=0.2)
argparser.add_argument("--num_of_samples", type=int, default=1)
argparser.add_argument("--cuda_device", type=int, default=1)
argparser.add_argument("--output_dir", type=str, default="./attr_edit_eval/ours_lora2_visual/")
argparser.add_argument("--prompt_file_path", type=str, default="./infer_images/comparison_attr_edit.txt")
args = argparser.parse_args()


WEIGHT_ID = args.weight_id
LORA_SCALE = args.lora_scale
NUM_OF_SAMPLES = args.num_of_samples
use_lora_finetuned = True
use_text_edits = False

# maske sure the program is stopped completely and not moves to next loop when ctrl+c is pressed


for i, identity in enumerate(all_id):
    file_name = identity
    identity = identity.split(".")[0]
    # if(i ==len(all_id)-1):
    #     break
    # filename2 = all_id[i+1]
    print("Genrating for identity: ", identity)
    if(not use_text_edits):
        if(use_lora_finetuned):
            subprocess.run(f'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_pmm_attr_eval.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} {identity} "0 0 0 0" True {NUM_OF_SAMPLES} {WEIGHT_ID} {LORA_SCALE} "" 0.0 "" 0.0 {file_name} {args.output_dir}', shell=True)
        else:
            subprocess.run(f'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_pmm_attr_eval.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} wt_interpolation_sd2_idloss "0 0 0 0" False {NUM_OF_SAMPLES} 149999 0.0 "" 0.0 "" 0.0 {file_name} {args.output_dir}', shell=True)
    else:
        if(use_lora_finetuned):
            subprocess.run(f'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_text_edit.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} {identity} "0 0 0 0" True {NUM_OF_SAMPLES} {WEIGHT_ID} {LORA_SCALE} "" 0.0 {file_name} {args.output_dir}', shell=True)
        else:    
            subprocess.run(f'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_text_edit.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} wt_interpolation_sd2_idloss "0 0 0 0" False {NUM_OF_SAMPLES} 149999 0.0 "" 0.0 {file_name} {args.output_dir}', shell=True)


    print("Done for identity: ", identity)

print("Done for all identities")