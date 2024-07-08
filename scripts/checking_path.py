import os
import shutil

saved_dir = "./outputs_all_ids_one/"

id_list = []
for folder in os.listdir(saved_dir):
    id_list.append(folder)

prompt_list = []
for prompt in os.listdir(os.path.join(saved_dir, id_list[0], "prompt_mixing/steps_10/sample_-1")):
    prompt_list.append(prompt)

prompt_list.pop(prompt_list.index("gif"))
print(prompt_list)

for id in id_list:
    print("id :", id)
    for prompt in prompt_list:
        shutil.move(os.path.join(saved_dir, id, "prompt_mixing/steps_10/sample_-1", prompt), \
                    os.path.join(saved_dir, id, prompt ))
        shutil.rmtree(os.path.join(saved_dir, id, "prompt_mixing"))

        # print("src path :", os.path.join(saved_dir, id, "prompt_mixing/steps_10/sample_-1", prompt))
        # print("dest path :", os.path.join(saved_dir, id, prompt ))   