import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn.functional as F

class ModelInferBase(ABC):
    def __init__(self, eval_out_dir: str, batch_size: int, device: torch.device,
                 opt: argparse.Namespace,
                 repeats: int = 1,
                 resume_cnt: int = 0,
                 ):
        os.makedirs(eval_out_dir, exist_ok=True)
        print('[ModelInferBase] All files will be saved to:\n {}'.format(eval_out_dir))
        self.eval_out_dir = eval_out_dir
        self.device = device
        self.batch_size = batch_size
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.repeats = repeats
        self.resume_cnt = resume_cnt

        self.opt = opt
        self.sampler = None
        self.start_code = None
        self.model = self._load_model(opt)

    @abstractmethod
    def _load_model(self, opt) -> nn.Module:
        pass

    @abstractmethod
    def infer_one(self, prompt: str, in_image: torch.Tensor, in_ids: torch.Tensor,
                  nums_id: torch.Tensor) -> torch.Tensor:
        """
        :param prompt: str
        :param in_image: (B,H,W,(1+diff+1+diff)C), only used by our method
        :param in_ids: (B,1+diff+1+diff), only used by our method
        :param nums_id: (B,), only used by our method
        :return (B,C,H,W)
        """

    def _img_path_list_to_bhwc(self, img_path_list: list):
        tensors = []
        for img_path in img_path_list:
            img = Image.open(img_path, 'r').convert('RGB')
            tensor = self.trans(img).permute(1, 2, 0)
            tensor = tensor.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)  # (B,H,W,C)
            tensor = tensor.to(self.device)
            tensors.append(tensor)
        return torch.cat(tensors, dim=-1)

    def _save_bchw_to_jpg(self, img_tensor: torch.Tensor, sub_folder: str):
        abs_folder = os.path.join(self.eval_out_dir, sub_folder)
        os.makedirs(abs_folder, exist_ok=True)
        img_cnt = len(os.listdir(abs_folder))
        img_tensor = torch.clamp((img_tensor + 1.0) / 2.0, min=0.0, max=1.0).permute(0, 2, 3, 1)  # (B,H,W,C)
        img_tensor *= 255.
        b, h, w, c = img_tensor.shape
        for i in range(b):
            a_array = img_tensor[i].cpu().numpy().astype(np.uint8)
            Image.fromarray(a_array).save(os.path.join(abs_folder, f"{img_cnt:05}.jpg"))
            img_cnt += 1

    def _save_list(self, a_list: list, name: str):
        a_list = ['{}\n'.format(x) for x in a_list]
        with open(os.path.join(self.eval_out_dir, name), 'w') as f:
            f.writelines(a_list)
        print(f'[ModelInferBase] {name} saved to {self.eval_out_dir}.')

    @torch.no_grad()
    def start_infer(self, prompts: list, in_image_paths: list, in_image_ids: list) -> list:
        self._save_list(prompts, 'prompts.txt')
        self._save_list(in_image_paths, 'in_image_paths.txt')
        self._save_list(in_image_ids, 'in_image_ids.txt')

        img_tensors = [self._img_path_list_to_bhwc(x) for x in in_image_paths]
        id_tensors = [torch.tensor(ids, device=self.device).unsqueeze(0).repeat(
            self.batch_size, 1) for ids in in_image_ids]
        nums_id_tensor = (torch.ones(self.batch_size, dtype=torch.long).to(self.device)) * 2
        out_tensors = []
        for i in tqdm(range(len(prompts)), desc='Synthesizing', position=1):
            if i < self.resume_cnt:
                continue
            prompt = [prompts[i]] * self.batch_size
            img_tensor = img_tensors[i]
            id_tensor = id_tensors[i]
            out_sub_folder = f'imgs/{i:05d}_id{in_image_ids[i][0]:05d}_{prompt[0]}'
            if i == self.resume_cnt and self.resume_cnt > 0:
                os.system(f'rm -r \'{os.path.join(self.eval_out_dir, out_sub_folder)}\'')

            for r in range(self.repeats):
                out_tensor = self.infer_one(prompt, img_tensor, id_tensor, nums_id_tensor)
                out_tensors.append(out_tensor)
                self._save_bchw_to_jpg(out_tensor, out_sub_folder)

            torch.cuda.empty_cache()
        print(f'[ModelInferBase] Synthesize finished, files saved to {self.eval_out_dir}.')
        return out_tensors


class EvalDatasetBase(ABC):
    def __init__(self, eval_dataset: str,
                 eval_ids: str,
                 prompts_file: str,
                 img_idx: int,
                 eval_id2s: str = "",
                 ):
        self.eval_dataset = eval_dataset
        self.eval_id1s = self._get_ids(eval_ids)
        self.eval_id2s = self._get_ids(eval_id2s, default=[])
        self.prompts_file = prompts_file
        self.img_idx = img_idx

        self._eval_meta_data = {}
        self.eval_data = self._construct_eval_data()

    @staticmethod
    def _get_ids(eval_ids: str, default=None):
        import re
        if re.match(r'\d+$', eval_ids) is not None:
            lo, hi = int(eval_ids), int(eval_ids) + 1
        elif re.match(r'\d+-\d+$', eval_ids) is not None:
            lo, hi = [int(x) for x in eval_ids.split('-')]
        else:
            if default is not None:
                return default
            raise ValueError('Specific_ids not supported.')
        return list(np.arange(lo, hi))

    def _construct_eval_data(self):
        if self.eval_dataset in ('vgg0', 'vgg1'):
            shift_id = 5
            total_id = 10
            imgs_per_id = 10
            dataset_path = "/gavin/datasets/aigc_id/dataset_{0}/".format(self.eval_dataset)
        elif self.eval_dataset in tuple([f'st{i}' for i in range(1, 11)]):
            shift_id = 1
            total_id = 10
            imgs_per_id = 1
            dataset_path = "/gavin/datasets/aigc_id/dataset_{0}/".format(self.eval_dataset)
        elif self.eval_dataset in ('e4t1',):
            shift_id = 4
            total_id = 7
            imgs_per_id = 1
            dataset_path = "/gavin/datasets/aigc_id/dataset_e4t/test/"
        else:
            raise ValueError('Eval dataset not supported:', self.eval_dataset)

        eval_id2s = [(x + shift_id) % total_id for x in self.eval_id1s]
        if len(self.eval_id2s) > 0:
            self.eval_id1s = np.array(self.eval_id1s)
            self.eval_id2s = np.array(self.eval_id2s)
            len1, len2 = len(self.eval_id1s), len(self.eval_id2s)
            self.eval_id1s = self.eval_id1s.repeat(len2)
            self.eval_id2s = np.tile(self.eval_id2s, len1)
            eval_id2s = self.eval_id2s
        print('[Eval id1]:', self.eval_id1s, '[Eval id2]:', eval_id2s)

        ''' ids '''
        in_image_ids = []
        for i in range(len(self.eval_id1s)):
            in_image_ids.append([self.eval_id1s[i], eval_id2s[i]])

        ''' in_image_paths (consistent with ids) '''
        in_image_paths = []
        for i in range(len(in_image_ids)):
            line = []
            for a_id in in_image_ids[i]:
                img_path = os.path.join(dataset_path,
                                        "{0:05d}_id{1}_#{2}.jpg".format(
                                            a_id * imgs_per_id + self.img_idx,
                                            a_id,
                                            self.img_idx
                                        ))
                if not os.path.exists(img_path):
                    raise FileNotFoundError('Not found:', img_path)
                line.append(img_path)
            in_image_paths.append(line)

        ''' prompts '''
        with open(self.prompts_file, 'r') as f:
            prompts = f.read().splitlines()

        self._eval_meta_data = {
            "prompts": prompts,  # [L]
            "in_image_paths": in_image_paths,  # [N*[2]]
            "in_image_ids": in_image_ids  # [N*[2]]
        }

        L = len(prompts)
        N = len(in_image_ids)
        eval_data = {
            "prompts": [],
            "in_image_paths": [],  # [N*[2]]
            "in_image_ids": []  # [N*[2]]
        }
        for i in range(N):
            in_image_path = in_image_paths[i]
            in_image_id = in_image_ids[i]
            for j in range(L):
                prompt = prompts[j]
                eval_data['prompts'].extend([prompt])
                eval_data['in_image_paths'].extend([in_image_path])
                eval_data['in_image_ids'].extend([in_image_id])
        print('[EvalDatasetBase] Eval data constructed. len:{}={}prompts*{}ids'.format(
            len(eval_data['prompts']),
            len(self._eval_meta_data['prompts']),
            len(self._eval_meta_data['in_image_ids'])
        ))
        return eval_data


class ImageSynthesizerBase(ABC):
    def __init__(self, eval_dataset: EvalDatasetBase,
                 generator: ModelInferBase,
                 ):
        self.eval_dataset = eval_dataset
        self.generator = generator

    def start_synthesize(self):
        self.generator.start_infer(
            self.eval_dataset.eval_data['prompts'],
            self.eval_dataset.eval_data['in_image_paths'],
            self.eval_dataset.eval_data['in_image_ids']
        )


class GeneratedDataset(Dataset):
    def __init__(self,
                 eval_folder: str,
                 prompt_file: str = './infer_images/comparison_prompt.txt',
                 in_image_paths_file: str = './aug_images/comparision/edited',
                 in_image_ids_file: str = 'in_image_ids.txt',
                 num_of_img_per_prompt: int = -1,
                 ):
        self.eval_folder = eval_folder
        self.prompt_file = prompt_file
        self.in_image_paths_file = in_image_paths_file
        self.in_image_ids_file = in_image_ids_file
        self.num_of_img_per_prompt = num_of_img_per_prompt

        self.prompts = []  # [L]
        self.src_img_paths = []  # [N*[2]]
        self.src_ids = []  # [N*[2]]
        self.gen_img_folders = []  # [L]
        self._load_txt_files()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((512, 512))
        ])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        src_img_path = str(self.src_img_paths[index][0])
        # src_id = self.src_ids[index]
        gen_img_folder = self.gen_img_folders[index]

        src_img_tensor = self._img_path_to_nchw(src_img_path)  # (1,C,H,W)
        gen_img_tensors = [self._img_path_to_nchw(
            os.path.join(gen_img_folder, x)
        ) for x in os.listdir(gen_img_folder)]  # (RB,C,H,W), R=repeats, B=batch_size
        if(self.num_of_img_per_prompt < 0):  
            gen_img_tensors = torch.stack(gen_img_tensors, dim=0) # (1,C,H,W)
        else:
            gen_img_tensors = torch.stack(gen_img_tensors, dim=0)[:self.num_of_img_per_prompt]  # (1,C,H,W)

        return {
            'prompt': prompt,
            'src_img_tensor': src_img_tensor,
            'gen_img_tensors': gen_img_tensors
        }

    def _load_txt_files(self):
        import re
        prompts_file = self.prompt_file
        in_image_paths_file = self.in_image_paths_file
        in_image_ids_file = 'in_image_ids.txt'
        identity_list = []
        for identity in sorted(os.listdir(self.eval_folder)):
            if os.path.isdir(os.path.join(self.eval_folder, identity)):
                identity_list.append(identity)

        src_img_list = sorted(os.listdir(in_image_paths_file))
        src_img_id = [id.split('.')[0] for id in src_img_list]
        evaluation_id_list = []
        for identity in identity_list:
            if identity not in src_img_id:
                print("Identity not found: ", identity, "skipping")
            else:
                evaluation_id_list.append(os.path.join(in_image_paths_file, src_img_list[src_img_id.index(identity)]))

        print("[Evaluation id list] :", len(evaluation_id_list), evaluation_id_list)
        if(len(identity_list) != len(src_img_id)):
            print("Length of eval folder: ", len(identity_list))
            print("Length of identity list: ", len(src_img_id))
            print("Length of eval folder != Length of src img list")
        
        with open(os.path.join(prompts_file), 'r') as f:
            prompt_once = f.read().splitlines()
            ##
            self.prompt_dict = {}
            for prompt in sorted(prompt_once):
                self.prompt_dict[prompt] = {"img_sim": [], "txt_sim": [], "id_cos_sim": [], "id_mse_dist": [], "id_l2_dist": [], 
                                                    "has_face": [], "no_face": []}
            prompts = []
            for i in range(len(identity_list)):
                for j in range(len(prompt_once)):
                    prompts.append(prompt_once[j])
            print("prompts length : ", len(prompts), prompts)
        # path_pattern = re.compile(r'[a-zA-Z\d#.:/_-]+')
        # with open(os.path.join(self.eval_folder, in_image_paths_file), 'r') as f:
        #     src_img_paths = f.read().splitlines()
        #     src_img_paths = [path_pattern.findall(line) for line in src_img_paths]
        #     print("src_img_paths length : ", len(src_img_paths))
            ##
        src_img_paths = []
        for identity in identity_list:
            for prompt in prompt_once:
                src_img_paths.append(
                    [os.path.join(in_image_paths_file, src_img_list[src_img_id.index(identity)])]
                )
        # num_pattern = re.compile(r'\d+')
        # with open(os.path.join(self.eval_folder, in_image_ids_file), 'r') as f:
        #     src_ids = f.read().splitlines()
        #     src_ids = [num_pattern.findall(line) for line in src_ids]

        gen_img_folders = []
        for identity in identity_list:
            for prompt in prompt_once:
                # gen_img_list = os.listdir(os.path.join(self.eval_folder, identity, prompt))
                gen_img_folders.append(
                    os.path.join(self.eval_folder,identity, prompt))

        self.prompts = prompts
        self.src_img_paths = src_img_paths
        # self.src_ids = src_ids
        self.gen_img_folders = gen_img_folders
        print('[GeneratedDataset] txt files loaded.')

    def _img_path_to_nchw(self, img_path: str):
        image = Image.open(img_path).convert('RGB')
        image = self.trans(image)
        return image

    @staticmethod
    def _img_paths_to_nchw(img_paths: str):
        images = [(np.array(Image.open(path).convert('RGB')) / 127.5 - 1.0).astype(np.float32) for path in img_paths]
        images = [torch.from_numpy(x).permute(2, 0, 1) for x in images]
        images = torch.stack(images, dim=0)  # (N,C,H,W)
        return images


class EvaluatorBase(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, gen: torch.Tensor, src: torch.Tensor, txt: str):
        pass


class IDCLIPScoreCalculator(ABC):
    def __init__(self,
                 eval_folder: str,
                 id_clip_evaluator: EvaluatorBase,
                 device: torch.device = torch.device('cuda:0'),
                 num_of_img_per_prompt: int = -1,
                 prompt_file: str = './infer_images/comparison_prompt.txt',
                 in_image_paths_file: str = './aug_images/comparision/edited',
                 ):
        self.eval_folder = eval_folder
        self.device = device
        self.num_of_img_per_prompt = num_of_img_per_prompt
        self.prompt_file = prompt_file
        self.in_image_paths_file = in_image_paths_file

        self.gen_dataset = None
        self.data_loader = None
        self._get_dataset_dataloader()

        self.id_clip_evaluator = id_clip_evaluator

    def _get_dataset_dataloader(self):
        gen_dataset = GeneratedDataset(
            self.eval_folder,
            prompt_file=self.prompt_file,
            num_of_img_per_prompt=self.num_of_img_per_prompt,
            in_image_paths_file=self.in_image_paths_file,
        )

        self.prompt_dict = gen_dataset.prompt_dict
        data_loader = DataLoader(
            gen_dataset, shuffle=False, batch_size=1, num_workers=4, drop_last=False
        )  # batch_size should be 1
        self.gen_dataset = gen_dataset
        self.data_loader = data_loader

    @torch.no_grad()
    def start_calc(self, include_undetected_samples: bool = False):
        sim_img_list = []
        sim_text_list = []
        id_cos_sim_list = []
        id_mse_dist_list = []
        id_l2_dist_list = []
        text_id_metrics = []
        sim_text_list_all_samples = []
        id_cos_sim_list_all_samples = []
        id_mse_dist_list_all_samples = []
        id_l2_dist_list_all_samples = []
        text_id_metrics_all_samples = []

        num_has_face, num_no_face = 0, 0
        for data in self.data_loader:
            src = data['src_img_tensor'].cuda()
            gen = data['gen_img_tensors'].squeeze(dim=0).cuda()
            prompt = data['prompt'][0]
            print("Prompt L : ", prompt)
            sim_img, sim_text, id_result_dict = self.id_clip_evaluator.evaluate(
                gen.repeat(1, 1, 1, 1),
                src.repeat(1, 1, 1, 1),
                prompt.replace('sks', ''))

            id_cos_sim = id_result_dict["cos_sim"]
            id_mse_dist = id_result_dict["mse_dist"]
            id_l2_dist = id_result_dict["l2_dist"]
            has_face = id_result_dict["num_has_face"]
            no_face = id_result_dict["num_no_face"]
            text_id_value = id_cos_sim * sim_text

            print("Image similarity: ", sim_img)
            print("Text similarity: ", sim_text)
            print("Identity cos similarity: ", id_cos_sim)

            # For plotting each prompt
            self.prompt_dict[prompt]["img_sim"].append(sim_img.cpu().numpy())
            self.prompt_dict[prompt]["txt_sim"].append(sim_text.cpu().numpy())
            self.prompt_dict[prompt]["id_cos_sim"].append(id_cos_sim if type(id_cos_sim) == int else id_cos_sim.cpu().numpy())
            self.prompt_dict[prompt]["id_mse_dist"].append(id_mse_dist)
            self.prompt_dict[prompt]["id_l2_dist"].append(id_l2_dist)
            self.prompt_dict[prompt]["has_face"].append(has_face)
            self.prompt_dict[prompt]["no_face"].append(no_face)

            print("has face : ", has_face, "no face : ", no_face)
            sim_img_list.append(sim_img)
            sim_text_list.append(sim_text)
            if id_cos_sim > 1e-6:
                sim_text_list_all_samples.append(sim_text)
                id_cos_sim_list.append(id_cos_sim)
                text_id_metrics.append(text_id_value)
                id_mse_dist_list.append(torch.FloatTensor([id_mse_dist]))
                id_l2_dist_list.append(torch.FloatTensor([id_l2_dist]))
                text_id_metrics_all_samples.append(text_id_value)
                id_cos_sim_list_all_samples.append(id_cos_sim)
                id_mse_dist_list_all_samples.append(torch.FloatTensor([id_mse_dist]))
                id_l2_dist_list_all_samples.append(torch.FloatTensor([id_l2_dist]))
            else:
                if(include_undetected_samples):
                    text_id_metrics_all_samples.append(torch.tensor(0.0).to(self.device))
                    sim_text_list_all_samples.append(torch.tensor(0.0).to(self.device))
                    id_cos_sim_list_all_samples.append(torch.tensor(0.0).to(self.device))
                    id_mse_dist_list_all_samples.append(torch.FloatTensor([id_mse_dist]))
                    id_l2_dist_list_all_samples.append(torch.FloatTensor([id_l2_dist]))
    
            num_has_face += has_face
            num_no_face += no_face

        # Getting text X Id metric(hm ,gm)
        txt_id_hm_metrics = []
        txt_id_gm_metrics = []
        txt_id_hm_metrics_all_samples = []
        txt_id_gm_metrics_all_samples = []
        for i in range(len(sim_text_list_all_samples)):
            id_cs = id_cos_sim_list_all_samples[i]
            txt_cs = sim_text_list_all_samples[i]
            if(id_cs != 0.0):
                txt_id_hm_metrics.append(2 * id_cs * txt_cs / (id_cs + txt_cs + 1e-6))
                txt_id_gm_metrics.append(torch.sqrt(id_cs * txt_cs))
                txt_id_hm_metrics_all_samples.append(2 * id_cs * txt_cs / (id_cs + txt_cs + 1e-6))
                txt_id_gm_metrics_all_samples.append(torch.sqrt(id_cs * txt_cs))
            else:
                txt_id_hm_metrics_all_samples.append(2 * id_cs * txt_cs / (id_cs + txt_cs + 1e-6))
                txt_id_gm_metrics_all_samples.append(torch.sqrt(id_cs * txt_cs))
    

        sim_img_avg = torch.stack(sim_img_list, dim=0)
        sim_text_avg = torch.stack(sim_text_list, dim=0)
        id_cos_sim_avg = torch.stack(id_cos_sim_list, dim=0)
        id_mse_dist_avg = torch.stack(id_mse_dist_list, dim=0)
        id_l2_dist_avg = torch.stack(id_l2_dist_list, dim=0)
        text_id_metrics_avg = torch.stack(text_id_metrics, dim=0)
        txt_id_hm_metrics_avg = torch.stack(txt_id_hm_metrics, dim=0)
        txt_id_gm_metrics_avg = torch.stack(txt_id_gm_metrics, dim=0)


        sim_text_avg_all_samples = torch.stack(sim_text_list_all_samples, dim=0)
        id_cos_sim_avg_all_samples = torch.stack(id_cos_sim_list_all_samples, dim=0)
        id_mse_dist_avg_all_samples = torch.stack(id_mse_dist_list_all_samples, dim=0)
        id_l2_dist_avg_all_samples = torch.stack(id_l2_dist_list_all_samples, dim=0)
        text_id_metrics_avg_all_samples = torch.stack(text_id_metrics_all_samples, dim=0)
        txt_id_hm_metrics_avg_all_samples = torch.stack(txt_id_hm_metrics_all_samples, dim=0)
        txt_id_gm_metrics_avg_all_samples = torch.stack(txt_id_gm_metrics_all_samples, dim=0)
        # import pdb;pdb.set_trace()

        # product = torch.dot(F.normalize(id_cos_sim_avg, dim=-1, p=2).float(), F.normalize(sim_text_avg, dim=-1, p=2).float())
        # product_all_samples = torch.dot(F.normalize(id_cos_sim_avg_all_samples, dim=-1, p=2).float(), F.normalize(sim_text_avg_all_samples, dim=-1, p=2).float())

        print("Image similarity (avg): ", sim_img_avg.mean())
        print("Text similarity (avg): ", sim_text_avg.mean())
        print("Text similarity (avg) all samples: ", sim_text_avg_all_samples.mean())
        print("Identity cos similarity (avg): ", id_cos_sim_avg.mean(),
              f"mse_dist={id_mse_dist_avg.mean():.4f}, l2_dist={id_l2_dist_avg.mean():.4f}",
              f"has_face={num_has_face}, no_face={num_no_face}")
        print("Identity cos similarity (avg) all samples: ", id_cos_sim_avg_all_samples.mean(),
                f"mse_dist={id_mse_dist_avg_all_samples.mean():.4f}, l2_dist={id_l2_dist_avg_all_samples.mean():.4f}")
        print("Text similarity x Identity cos similarity (avg): ", text_id_metrics_avg.mean())
        print("Text similarity x Identity cos similarity (avg) all samples: ", text_id_metrics_avg_all_samples.mean())
        # print("Text similarity x Identity cos similarity (avg) normalized: ", product.mean())
        # print("Text similarity x Identity cos similarity (avg) all samples normalized: ", product_all_samples.mean())
        print("Text similarity x Identity cos similarity (avg) harmonic mean: ", txt_id_hm_metrics_avg.mean())
        print("Text similarity x Identity cos similarity (avg) geometric mean: ", txt_id_gm_metrics_avg.mean())
        print("Text similarity x Identity cos similarity (avg) all samples harmonic mean: ", txt_id_hm_metrics_avg_all_samples.mean())
        print("Text similarity x Identity cos similarity (avg) all samples geometric mean: ", txt_id_gm_metrics_avg_all_samples.mean())
        
        return self.prompt_dict
