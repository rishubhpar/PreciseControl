import os
import subprocess
import argparse
import yaml

all_id_path = "./aug_images/bindi_lora_data/"

all_id = sorted(os.listdir(all_id_path))
print("all_id: ", all_id)
# all_id = all_id[len(all_id)//2:]
# all_id = ["gates.jpg", "ed.jpg","lecun.jpg","bengio.jpg","fei.jpg","hinton.jpg"]
# all_id = ["morgan.jpg"]
# all_id = ["adele.jpg","snoop.jpg","feynman1.jpg","einstein1.jpg"]

# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 CUDA_VISIBLE_DEVICES=0 bash ./02_start_test_seed_sweep.sh "./weights/v2-1_512-ema-pruned.ckpt" "./infer_images/comparison_prompt.txt" "cnr" "0 0 0 0" True 4 49 0.3


argparser = argparse.ArgumentParser()
# argparser.add_argument("--weight_id", type=int, default=49)
# argparser.add_argument("--lora_scale", type=float, default=0.4)
# argparser.add_argument("--num_of_samples", type=int, default=4)
argparser.add_argument("--cuda_device", type=int, default=0)
argparser.add_argument("--yaml_file", type=str, default="./configs/stable-diffusion/aigc_id_for_lora_all.yaml")
args = argparser.parse_args()


# maske sure the program is stopped completely and not moves to next loop when ctrl+c is pressed

# check lora_finetuning file to see which model is used before training !!!!!!!!!!!!!!
for i, identity in enumerate(all_id):
    folder_name = identity
    
    with open(args.yaml_file, 'r') as file:
        filedata = yaml.load(file, Loader=yaml.SafeLoader)
    
    data_path = filedata["data"]["params"]["train"]["params"]["root_dir"]
    path_list = data_path.split("/")
    print("path_list: ", path_list)
    path_list[-2] = folder_name
    data_path = "/".join(path_list)
    print("data_path: ", data_path)
    filedata["data"]["params"]["train"]["params"]["root_dir"] = data_path
    filedata["data"]["params"]["validation"]["params"]["root_dir"] = data_path

    with open(args.yaml_file, 'w') as file:
        yaml.dump(filedata, file, default_flow_style=False, sort_keys=False)
    
    subprocess.run(f'CUDA_VISIBLE_DEVICES={args.cuda_device} bash ./01_start_lora_finetuning.sh ./weights/v2-1_512-ema-pruned.ckpt {folder_name}', shell=True)
    
    os.system(f"rm ./logs_makeup/{folder_name}/checkpoints/embeddings.pt")
    os.system(f"rm ./logs_makeup/{folder_name}/checkpoints/lora_params.pt")

    # os.system(f"rm ./logs_ravdess/{folder_name}/checkpoints/embeddings_gs-19.pt")
    # os.system(f"rm ./logs_ravdess/{folder_name}/checkpoints/lora_params-19.pt")
    # os.system(f"rm ./logs_ravdess/{folder_name}/checkpoints/embeddings_gs-29.pt")
    # os.system(f"rm ./logs_ravdess/{folder_name}/checkpoints/lora_params-29.pt")
    # os.system(f"rm ./logs_ravdess/{folder_name}/checkpoints/embeddings_gs-39.pt")
    # os.system(f"rm ./logs_ravdess/{folder_name}/checkpoints/lora_params-39.pt")

    print("Done for identity: ", identity)

print("Done for all identities")