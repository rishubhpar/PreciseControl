import os
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np
import argparse
from transformers import CLIPProcessor, CLIPModel
from lpips import LPIPS

# for clip score calculation
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# for lpips score calculation
# lpips_model = LPIPS(net='alex')
lpips_model = LPIPS(net='vgg').cuda()
lpips_image_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

parser = argparse.ArgumentParser()
parser.add_argument('--eval_img_dir', type=str, default='./attr_edit_eval/ip2p_edits/')
parser.add_argument('--attr_list', type=str, default=['smile'])
parser.add_argument('--src_img_dir', type=str, default='./aug_images/comparision/edited')
args = parser.parse_args()
# "a photo of face of a old person"
# "a photo of face of a person smiling"
# "a photo of face of a person with beard"
# "a photo of face of a asian person"
# "a photo of face of a person with black skin tone"

edit_prompt = "a photo of face of a person smiling"

def main(args):
    edits_img_dir = os.path.join(args.eval_img_dir)
    identity_list = [identity.split('.')[0] for identity in sorted(os.listdir(args.src_img_dir))]
    if(len(identity_list) != len(os.listdir(args.eval_img_dir))):
        print('Error: number of identities in edited images and source images are not equal')
        print("number of ids in eval folder : ", len(os.listdir(args.eval_img_dir)))
        print("number of ids in src folder : ", len(os.listdir(args.src_img_dir)))


    for attr in args.attr_list:
        clip_score_list = []
        lpips_score_list = []
        for identity_file in os.listdir(args.src_img_dir):
            identity = identity_file.split('.')[0]
            scr_img = Image.open(os.path.join(edits_img_dir, identity, "original", os.listdir(os.path.join(edits_img_dir, identity, "original"))[0])).convert('RGB')
            if(len(os.listdir(os.path.join(edits_img_dir, identity, attr)))!=1):
                print("more than one edit file presnt at ", os.path.join(edits_img_dir, identity, attr))
            edited_img = Image.open(os.path.join(edits_img_dir, identity, attr, os.listdir(os.path.join(edits_img_dir, identity, attr))[0])).convert('RGB')
            
            # lpips score calculation
            scr_img_tensor = lpips_image_transform(scr_img).unsqueeze(0).cuda()
            edited_img_tensor = lpips_image_transform(edited_img).unsqueeze(0).cuda()
            lpips_score = lpips_model(scr_img_tensor, edited_img_tensor).item()
            lpips_score_list.append(lpips_score)

            # clip score calculation
            scr_img_clip_input = processor(text=[edit_prompt], images=scr_img, return_tensors="pt", padding=True)
            edited_img_clip_input = processor(text=[edit_prompt], images=edited_img, return_tensors="pt", padding=True)
            # print("edit prompt : ", edit_prompt.format(attr))
            scr_clip_score = model(**scr_img_clip_input).logits_per_image
            # print("scr_clip_score : ", scr_clip_score)
            scr_clip_score = scr_clip_score[0][0].detach().cpu().numpy()
            # print("scr_clip_score : ", scr_clip_score)

            edited_clip_score = model(**edited_img_clip_input).logits_per_image
            # print("edited_clip_score : ", edited_clip_score)
            edited_clip_score = edited_clip_score[0][0].detach().cpu().numpy()
            # print("edited_clip_score : ", edited_clip_score)

            delta_clip_score = abs(scr_clip_score - edited_clip_score)
            # print("delta_clip_score : ", delta_clip_score)
            clip_score_list.append(delta_clip_score)
        
        print("clip score list : ", clip_score_list)
        print("lpips score list : ", lpips_score_list)
        print("clip score list mean : ", np.mean(clip_score_list))
        print("lpips score list mean : ", np.mean(lpips_score_list))


main(args)
