import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
sys.path.append('/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/')

import torch
from ldm.modules.e4e.psp import pSp
from ldm.modules.e4e.alignment import align_face
import dlib
import json
from PIL import Image


class E4EEncoder(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super(E4EEncoder, self).__init__()
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts["test_batch_size"] = 1
        # print(opts)

        opts['checkpoint_path'] = checkpoint_path
        opts['device'] = 'cuda'
        opts = argparse.Namespace(**opts)

        self.e4e_encoder = pSp(opts)
        self.e4e_encoder.eval()
        self.e4e_encoder.cuda()
        self.shape_predictor = dlib.shape_predictor('./weights/encoder/shape_predictor_68_face_landmarks.dat')

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def get_wlatents(self, x, is_cars=False):
        codes = self.e4e_encoder.encoder(x)
        self.e4e_encoder.latent_avg = self.e4e_encoder.latent_avg.to(codes.device)
        if self.e4e_encoder.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.e4e_encoder.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.e4e_encoder.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and is_cars:
            codes = codes[:, :16, :]
        return codes
    

def get_delta_w(encoder, x1, x2, is_cars=False):
    w1 = encoder.get_wlatents(x1, is_cars=is_cars)
    w2 = encoder.get_wlatents(x2, is_cars=is_cars)
    delta_w = w2 - w1
    return delta_w


dataset_path = './aug_images/flame_pair/'
delta_w_dict = {}

encoder = E4EEncoder('/mnt/data/rishubh/sachi/CelebBasis_pstar/weights/encoder/e4e_ffhq_encode.pt')

for folder in os.listdir(dataset_path):
    if(folder != 'pose'):
        folder_path = os.path.join(dataset_path, folder)
        files_name = os.listdir(folder_path)
        ids = [files_name.split("_")[0] for files_name in files_name]
        ids = list(set(ids))
        delta_w_list = []
        for id in ids:
            align_x1 = align_face(os.path.join(folder_path, f'{id}_0.jpg'), encoder.shape_predictor)
            align_x2 = align_face(os.path.join(folder_path, f'{id}_1.jpg'), encoder.shape_predictor)
            x1 = encoder.transform(align_x1).unsqueeze(0).cuda()
            x2 = encoder.transform(align_x2).unsqueeze(0).cuda()

            delta_w = get_delta_w(encoder, x1, x2).squeeze(0).reshape(-1)
            delta_w_list.append(delta_w.cpu().detach().numpy())
        delta_w = np.mean(np.array(delta_w_list), axis=0)
        delta_w = delta_w.reshape(18, -1)
        print("delta_w: ", delta_w.shape)
        delta_w_dict[folder] = delta_w.tolist()
    else:
        folder_path = os.path.join(dataset_path, folder)
        files_name = os.listdir(folder_path)
        ids = [files_name.split("_")[0] for files_name in files_name]
        ids = list(set(ids))
        delta_w_list = []
        for id in ids:
            align_x1 = align_face(os.path.join(folder_path, f'{id}_0.jpg'), encoder.shape_predictor)
            align_x2 = align_face(os.path.join(folder_path, f'{id}_1.jpg'), encoder.shape_predictor)
            x1 = encoder.transform(align_x1).unsqueeze(0).cuda()
            x2 = encoder.transform(align_x2).unsqueeze(0).cuda()

            delta_w = get_delta_w(encoder, x1, x2).squeeze(0).reshape(-1)
            delta_w_list.append(delta_w.cpu().detach().numpy())
        delta_w = np.mean(np.array(delta_w_list), axis=0)
        delta_w = delta_w.reshape(18, -1)
        print("delta_w: ", delta_w.shape)
        delta_w_dict[folder] = delta_w.tolist()

json.dump(delta_w_dict, open('flame100_delta_w_dict.json', 'w'))

    