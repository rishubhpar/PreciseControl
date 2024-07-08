import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
sys.path.append("..")

import clip
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluation.base_class_attr_edit import IDCLIPScoreCalculator
from evaluation.clip_eval1 import IdCLIPEvaluator
from evaluation.parse_args import parser_eval

plot_values = False


if __name__ == "__main__":
    """
    Usage:
    export PYTHONPATH=/gavin/code/TextualInversion/
    python evaluation/eval_imgs.py --eval_out_dir ./exp_eval/cd  \
        --eval_project_folder 2023-05-05T18-42-56_two_person-sdv4  \
        --eval_time_folder eval_2023-05-17T15-22-32
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder", type=str, default="./attr_edit_eval/ip2p_edits/") # ./benchmark_results/ours_lora2
    parser.add_argument("--eval_out_dir", type=str, default="./exp_eval/ours/")
    parser.add_argument("--eval_project_folder", type=str, default="2023-05-05T18-42-56_two_person-sdv4")
    parser.add_argument("--num_of_img_per_prompt", type=int, default=1)
    parser.add_argument("--prompt_file", type=str, default="./infer_images/comparison_attr_edit.txt")
    parser.add_argument("--include_no_face_detect_samples", type=bool, default=True)
    args = parser.parse_args()

    attr_list = ["smile"]

    for attr in attr_list:
        id_clip_evaluator = IdCLIPEvaluator(
            torch.device('cuda:0'),
        )
        if(not plot_values):
            id_score_calculator = IDCLIPScoreCalculator(
                args.eval_folder,
                id_clip_evaluator,
                num_of_img_per_prompt=args.num_of_img_per_prompt,
                prompt_file=args.prompt_file,
                attr = attr,
            )

            _ = id_score_calculator.start_calc(args.include_no_face_detect_samples)

        else:
            # For each id folder, calculate the score
            save_hist_plot_dir = "./out_histplot_custom_diff/"
            shutil.rmtree(save_hist_plot_dir, ignore_errors=True)
            os.makedirs(os.path.join(save_hist_plot_dir), exist_ok=True)
            root_output_dir = args.eval_folder

            img_sim_across_all_id = []
            text_sim_across_all_id = []
            id_cos_sim_across_all_id = []
            id_mse_dist_across_all_id = []
            id_l2_dist_across_all_id = []

            # create a imaginary dir and copy one id folder there and make the eval_folder and at end of loop delete it
            imag_dir = "./imaginary_dir/"
            if(os.path.exists(imag_dir)):
                shutil.rmtree(imag_dir)
            os.makedirs(os.path.join(imag_dir), exist_ok=True)

            all_id = sorted(os.listdir(root_output_dir))
            for identity in tqdm(all_id):
                if(identity == "einstein" or identity == "altman"):
                    continue
                identity_dir = os.path.join(root_output_dir, identity)
                if not os.path.isdir(identity_dir):
                    continue

                shutil.copytree(identity_dir, os.path.join(imag_dir, identity))
                
                print("Calculating score for identity: ", identity)
                id_score_calculator = IDCLIPScoreCalculator(
                    imag_dir,
                    id_clip_evaluator,
                    num_of_img_per_prompt=args.num_of_img_per_prompt,
                )

                prompt_dict = id_score_calculator.start_calc()

                hist_across_prompt = []
                sim_across_prompt = []
                for prompt, prompt_value in prompt_dict.items():
                    hist_across_prompt.append(np.array(prompt_value["txt_sim"]).mean())
                    sim_across_prompt.append(np.array(prompt_value["id_cos_sim"]).mean())
                
                text_sim_across_all_id.append(hist_across_prompt)
                id_cos_sim_across_all_id.append(sim_across_prompt)

                identity_dir2 = os.path.join("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/benchmark_results/celebbasis/", identity)
                if not os.path.isdir(identity_dir2):
                    continue
                shutil.rmtree(os.path.join(imag_dir, identity))
                shutil.copytree(identity_dir2, os.path.join(imag_dir, identity))

                print("Calculating score for identity: ", identity)
                id_score_calculator = IDCLIPScoreCalculator(
                    imag_dir,
                    id_clip_evaluator,
                    num_of_img_per_prompt=args.num_of_img_per_prompt,
                )
                prompt_dict2 = id_score_calculator.start_calc()

                hist_across_prompt2 = []
                sim_across_prompt2 = []
                for prompt, prompt_value in prompt_dict2.items():
                    hist_across_prompt2.append(np.array(prompt_value["txt_sim"]).mean())
                    sim_across_prompt2.append(np.array(prompt_value["id_cos_sim"]).mean())
                
                # plot the bar for this identity with prompts at x axis for each bar
                # change size to 20, 20
                plt.figure(figsize=(20, 20))
                plt.bar(list(prompt_dict.keys()), hist_across_prompt, width=0.4, label="ours")
                plt.bar(list(prompt_dict2.keys()), hist_across_prompt2, width=0.1, label="custom_diffusion")
                plt.xticks(rotation=90)
                plt.title(identity)
                plt.xlabel("Prompt")
                plt.ylabel("Text similarity")
                plt.ylim(0, 1)
                plt.savefig(os.path.join(save_hist_plot_dir, identity+"_text_sim.png"))
                plt.close()

                plt.figure(figsize=(20, 20))
                plt.bar(list(prompt_dict.keys()), sim_across_prompt, width=0.4, label="ours")
                plt.bar(list(prompt_dict2.keys()), sim_across_prompt2, width=0.1, label="custom_diffusion")
                plt.xticks(rotation=90)
                plt.title(identity)
                plt.xlabel("Prompt")
                plt.ylabel("ID cosine similarity")
                plt.ylim(0, 1)
                plt.savefig(os.path.join(save_hist_plot_dir, identity+"_id_cos_sim.png"))
                plt.close()
                print("Done for identity: ", identity)

                shutil.rmtree(os.path.join(imag_dir, identity))

            if(os.path.exists(imag_dir)):
                shutil.rmtree(imag_dir)
            
            # plot across all identities
            text_sim_across_all_id = np.array(text_sim_across_all_id)
            id_cos_sim_across_all_id = np.array(id_cos_sim_across_all_id)
            text_sim_across_all_id = text_sim_across_all_id.mean(axis=0)
            id_cos_sim_across_all_id = id_cos_sim_across_all_id.mean(axis=0)
            print("num of identities: ", len(text_sim_across_all_id))
            plt.figure(figsize=(20, 20))
            plt.bar(list(prompt_dict.keys()), np.array(text_sim_across_all_id).mean(axis=0))
            plt.xticks(rotation=90)
            plt.title("all_id")
            plt.xlabel("Prompt")
            plt.ylabel("Text similarity")
            plt.ylim(0, 1)
            plt.savefig(os.path.join(save_hist_plot_dir, "all_id_text_sim.png"))
            plt.close()


        #     # plot the bar for this identity with prompts at x axis for each bar
        #     # change size to 20, 20
        #     plt.figure(figsize=(20, 20))
        #     plt.bar(list(prompt_dict.keys()), hist_across_prompt)
        #     plt.xticks(rotation=90)
        #     plt.title(identity)
        #     plt.xlabel("Prompt")
        #     plt.ylabel("Text similarity")
        #     plt.ylim(0, 1)
        #     plt.savefig(os.path.join(save_hist_plot_dir, identity+"_text_sim.png"))
        #     plt.close()

        #     plt.figure(figsize=(20, 20))
        #     plt.bar(list(prompt_dict.keys()), sim_across_prompt)
        #     plt.xticks(rotation=90)
        #     plt.title(identity)
        #     plt.xlabel("Prompt")
        #     plt.ylabel("ID cosine similarity")
        #     plt.ylim(0, 1)
        #     plt.savefig(os.path.join(save_hist_plot_dir, identity+"_id_cos_sim.png"))
        #     plt.close()
        #     print("Done for identity: ", identity)

        #     shutil.rmtree(os.path.join(imag_dir, identity))

        # if(os.path.exists(imag_dir)):
        #     shutil.rmtree(imag_dir)
                


