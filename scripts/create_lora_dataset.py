import os
import shutil

# root_dataset_path = "./aug_images/comparision_new/edited/"

# id_list = os.listdir(root_dataset_path)

root_dataset_path = "./aug_images/real_imgs/edited"

id_list = os.listdir(root_dataset_path)

for i, id in enumerate(id_list):
    name = id.split(".")[0]
    os.makedirs("./aug_images/real_lora_data/"+f"{name}"+"/0000", exist_ok=True)
    shutil.copy(os.path.join(root_dataset_path,id), "./aug_images/real_lora_data/"+f"{name}"+"/0000/"+id)

# print("Done")
    
