import os
import subprocess
import argparse

all_id_path = "./aug_images/bindi_imgs/edited/"

all_id = sorted(os.listdir(all_id_path))[3:4]
print("all_id: ", all_id) 
# all_id = all_id[23:]
# all_id = ["gates.jpg", "ed.jpg","lecun.jpg","bengio.jpg","fei.jpg","hinton.jpg"]
# all_id = ["morgan.jpg"]
# all_id = ["adele.jpg","snoop.jpg","feynman1.jpg","einstein1.jpg"]

# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES=0 bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" "./infer_images/comparison_prompt.txt" "cnr" "0 0 0 0" True 4 49 0.3


argparser = argparse.ArgumentParser()
argparser.add_argument("--weight_id", type=int, default=49)
argparser.add_argument("--lora_scale", type=float, default=0.5)
argparser.add_argument("--num_of_samples", type=int, default=5)
argparser.add_argument("--cuda_device", type=int, default=2)
argparser.add_argument("--output_dir", type=str, default="./outputs_makeup05")
argparser.add_argument("--log_dir", type=str, default="logs_makeup")
argparser.add_argument("--input_img_folder", type=str, default="bindi_imgs")
argparser.add_argument("--prompt_file_path", type=str, default="./infer_images/comparison_prompt.txt")
args = argparser.parse_args()


WEIGHT_ID = args.weight_id
LORA_SCALE = args.lora_scale
NUM_OF_SAMPLES = args.num_of_samples
use_lora_finetuned = True
two_ids = False

# maske sure the program is stopped completely and not moves to next loop when ctrl+c is pressed


for i, identity in enumerate(all_id):
    file_name = identity
    identity = identity.split(".")[0]
    if(i ==len(all_id)-1 and two_ids):
        break
    if(two_ids):
        filename2 = all_id[i+1]
    print("Genrating for identity: ", identity)
    # if(identity != "jitendra"):
    #     continue
    if(not two_ids):
        if(use_lora_finetuned):
            subprocess.run(f'CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} {identity} "0 0 0 0" True {NUM_OF_SAMPLES} {WEIGHT_ID} {LORA_SCALE} {file_name} {args.output_dir} {args.log_dir} {args.input_img_folder}', shell=True)
        else:
            subprocess.run(f'CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} abalation_w_one_token "0 0 0 0" False {NUM_OF_SAMPLES} 149999 0.0 {file_name} {args.output_dir} {args.log_dir} {args.input_img_folder}', shell=True)
    else:
        if(use_lora_finetuned):
            subprocess.run(f'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} {identity} "0 0 0 0" True {NUM_OF_SAMPLES} {WEIGHT_ID} {LORA_SCALE} {file_name} {args.output_dir} {filename2}', shell=True)
        else:    
            subprocess.run(f'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" {args.prompt_file_path} wt_interpolation_sd2_idloss "0 0 0 0" False {NUM_OF_SAMPLES} 149999 0.0 {file_name} {args.output_dir} {filename2}', shell=True)


    print("Done for identity: ", identity)

print("Done for all identities")