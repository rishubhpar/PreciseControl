import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from PIL import Image
import pickle
import cv2
from torchvision.transforms import transforms

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from evaluation.prompt_templates import get_pos_neg_temps
from src.lora.lora_diffusion.lora import inject_trainable_lora, monkeypatch_or_replace_lora, tune_lora_scale


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, embedding_path, use_lora_finetuned, lora_scale, verbose=False):
    print(f"Loading model from {ckpt}")
    _, extension = os.path.splitext(ckpt)
    if extension.lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if((config.model.params.use_lora_finetuning == True) and use_lora_finetuned):
        # _ , _ = inject_trainable_lora(model.model, loras=embedding_path.replace("embeddings_gs", "lora_params"))
        print("[Injecting lora weights] from : ", embedding_path.replace("embeddings_gs", "lora_params"))
        print("[Lora rank, lora scale] : ", config.model.params.lora_rank, lora_scale)
        names_list = config.model.params.lora_names_list if hasattr(config.model.params, "lora_names_list") else None
        monkeypatch_or_replace_lora(model.model, loras=torch.load(embedding_path.replace("embeddings_gs", "lora_params")),r=config.model.params.lora_rank,
                                    names_list=names_list)
        tune_lora_scale(model.model, lora_scale)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    # Change this
    def list_of_ints(arg):
        return list(map(int, arg.split(' ')))
    
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_false',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        # default="configs/stable-diffusion/v1-inference.yaml",
        default="configs/stable-diffusion/aigc_id_infer.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--lora_finetuned",
        type=bool,
        default=False,
        help="whether the model is lora finetuned or not",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="lora weights scaling factor",
    )
    parser.add_argument(
        "--lora2_path",
        type=str,
        default='',
        help="Path for second id lora weights, if it is used",
    )
    parser.add_argument(
        "--lora2_scale",
        type=float,
        default=1.0,
        help="lora weights scaling factor",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default='',
        help="file name for saving",
    )
    parser.add_argument(
        "--id2_file_name",
        type=str,
        default="",
        help="second id file name",
    )
    parser.add_argument(
        "--inp_img_folder",
        type=str,
        default="",
        help="folder which has the image to be evaluated"
    )

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default='vgg1',
        help="Test target dataset. (for test_mode='image')")
    parser.add_argument(
        "--eval_folder",
        type=str,
        default='test',
        help="Test target folder. (for test_mode='image')")
    parser.add_argument(
        "--eval_id1",
        type=int,
        default=0,
        help="The id of the first person.")
    parser.add_argument(
        "--eval_id2",
        type=int,
        default=-1,
        help="The id of the second person.")
    parser.add_argument(
        "--eval_img_idx",
        type=int,
        default=0,
        help="The image idx of the person. (for test_mode='image')")
    parser.add_argument(
        "--img_suffix",
        type=str,
        default="",
        help="The suffix of saved images.")
    # get interpolate ids argument with type as list of int
    parser.add_argument("--interpolate_ids", type=list_of_ints, default=[0],
                        help="The ids to interpolate. structure [id1, id2, output interpolation number, total interpolation needed]]]")

    opt = parser.parse_args()
    print("Interpolate ids:", opt.interpolate_ids)

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", opt.embedding_path, opt.lora_finetuned, opt.lora_scale)
    model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    prompt_all = []
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        prompt_all.append(prompt)
        temp_neg = ""
    else:
        print(f"reading prompts from {opt.from_file}")
        # temp_pos, temp_neg = get_pos_neg_temps(opt.from_file)  # optional: use long prompt
        # temp_pos, temp_neg = "{}", "magzine, frame, tiled, repeated, multiple people, multiple faces, split frame, multiple panel, \
        #     split image, boarder, diptych, triptych, black and white, low quality, blurry image, out of frame, poorly rendered face, jpeg artifacts"
        # temp_pos, temp_neg = "{}", "3D, deformed, blurry, bad atonomy, disfigured, distorted, deformed, bad art, boring, low quality, poorly rendered, blurry, out of focus\
        #     out of frame, split frame, multiple panel, split image, frame, magazine, tiled, diptych, triptych,low resolution, distored face, cartoon, duplicate, repeated,  multiple faces, bad eyes, bad face, cropped head, black and white, \
        #         plain background, low details, distorted detail, unattractive, ugly, jpeg artifacts"
        temp_pos, temp_neg = "{}", "3D, deformed, diptych, triptych, blurry, bad atonomy, disfigured, distorted, deformed, bad art, boring, low quality, poorly rendered, blurry, out of focus\
            out of frame, low resolution, distored face, cartoon, duplicate, repeated,  multiple faces, bad eyes, bad face, cropped head, black and white, \
                plain background, low details, distorted detail, unattractive, ugly, jpeg artifacts, \
                split frame, multiple panel, split image, frame, magazine, tiled, diptych, triptych"
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            prompt_all = data

            data = [batch_size * [temp_pos.format(p)] for p in data]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    os.rmdir(sample_path)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if config.model.params.personalization_config.params.test_mode == "image":
        img_idx = opt.eval_img_idx
        if opt.eval_dataset in ('vgg0', 'vgg1'):
            shift_id = 5
            test_id2 = (opt.eval_id1 + shift_id) % 10
            id1 = Image.open("/gavin/datasets/aigc_id/dataset_{0}/{1:05d}_id{2}_#{3}.jpg".format(
                opt.eval_dataset, opt.eval_id1 * 10 + img_idx, opt.eval_id1, img_idx)).convert("RGB")
            id2 = Image.open("/gavin/datasets/aigc_id/dataset_{0}/{1:05d}_id{2}_#{3}.jpg".format(
                opt.eval_dataset, test_id2 * 10 + img_idx, test_id2, img_idx)).convert("RGB")
        elif opt.eval_dataset in ('st1', 'st2'):
            shift_id = 1
            test_id2 = (opt.eval_id1 + shift_id) % 10
            id1 = Image.open("/gavin/datasets/aigc_id/dataset_{0}/{1:05d}_id{1}_#0.jpg".format(
                opt.eval_dataset, opt.eval_id1)).convert("RGB")
            id2 = Image.open("/gavin/datasets/aigc_id/dataset_{0}/{1:05d}_id{1}_#0.jpg".format(
                opt.eval_dataset, test_id2)).convert("RGB")
        elif opt.eval_dataset in ('e4t1', ):
            shift_id = 4
            test_id2 = (opt.eval_id1 + shift_id) % 7
            id1 = Image.open("/gavin/datasets/aigc_id/dataset_e4t/test/{0:05d}_id{0}_#0.jpg".format(
                opt.eval_id1)).convert("RGB")
            id2 = Image.open("/gavin/datasets/aigc_id/dataset_e4t/test/{0:05d}_id{0}_#0.jpg".format(
                test_id2)).convert("RGB")
        elif opt.eval_dataset in ('ffhq', ):

            # opt.eval_id1 = np.random.randint(len(os.listdir("/home/test/rishubh/sachi/CelebBasisV2/aug_images/stylegan3/edited/"))-1)
            # test_id2 = (opt.eval_id1 + 1) % 10
            test_id2 = 0
            # id1 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/stylegan3/edited/{}".format(
            #     os.listdir("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/stylegan3/edited/")[opt.eval_id1])).convert("RGB")
            # id2 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/stylegan3/edited/{}".format(
            #     os.listdir("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/stylegan3/edited/")[test_id2])).convert("RGB")
            # id1 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/comparision/edited/{}".format(
            #     os.listdir("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/comparision/edited/")[opt.eval_id1])).convert("RGB")
            # id2 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/comparision/edited/{}".format(
            #     os.listdir("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/comparision/edited/")[test_id2])).convert("RGB")

        else:
            raise ValueError('Eval dataset not supported:', opt.eval_dataset)
        trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        def pil_to_4d(img: Image):
            tensor = trans(img).permute(1, 2, 0)
            tensor = tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (N,H,W,C)
            return tensor.to(device)
        
        # two_ids = np.concatenate([np.array(id1), np.array(id2)], axis=1)
        # Image.fromarray(two_ids).save("./two_ids.jpg")  # for debug
        
        # changed for new mapper
        # faces = torch.cat([pil_to_4d(id1), pil_to_4d(id2)], dim=-1)
        # faces = pil_to_4d(id1)
    else:
        faces = None
        test_id2 = opt.eval_id1  # default

    # if opt.eval_id2 >= 0:
    #     test_id2 = opt.eval_id2


    import json
    delta_w_dict = json.load(open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/delta_w_dict.json"))
    # delta_w_dict = json.load(open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/latent_directions_flame.json"))
    # flame dict keys: face_{bald,bang,beard,eye_g,hat,lights,pose,smile}, open_close
    # attr = 'pose'
    attr = None
    weight_lambda = 10
    use_promt_mixing = True
    steps_for_prompt_mixing = 10
    seed_list = range(3,5,1)
    two_ids = True if opt.id2_file_name != "" else False
    multi_face = [opt.id2_file_name]
    print("multi face :", multi_face)

    if(use_promt_mixing):
        print("output path :", outpath)
        outpath = os.path.join(outpath, opt.file_name.split(".")[0])
        # outpath = os.path.join(outpath, "prompt_mixing", "steps_{}".format(steps_for_prompt_mixing))
        os.makedirs(outpath, exist_ok=True)
    print("result will be saved at ", outpath)
 
    grid_count = len(os.listdir(outpath)) - 1
    outpath_root = outpath
    # outpath_root = os.path.join(outpath, f"sample_{grid_count}")
    # os.makedirs(os.path.join(outpath_root,"gif"), exist_ok=True)
    grid_count = len(os.listdir(outpath_root)) 

    prompts_idx = 0
    for prompts in tqdm(data, desc="data"):
        outpath = os.path.join(outpath_root, str(prompts[0]))
        os.makedirs(outpath, exist_ok=True)
        grid_count = len(os.listdir(outpath))
        img_folder_name = opt.inp_img_folder
        for l, img in enumerate(os.listdir("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/{}/edited/".format(img_folder_name))):
            if(img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg")):
                print("reading file")
                # img = 'bengio.jpg'
                img = opt.file_name
                id1 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/{}/edited/{}".format(img_folder_name,img)).convert("RGB")
                faces = pil_to_4d(id1)

                if(two_ids):
                    multi_faces_list = [pil_to_4d(id1)]
                    for i in range(len(multi_face)):
                        id2 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/{}/edited/{}".format("comparision_new", multi_face[i])).convert("RGB")
                        multi_faces_list.append(pil_to_4d(id2))
                    # id2 = Image.open("/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/aug_images/{}/edited/{}".format("comp", "1.jpg")).convert("RGB")
                    # face2 = pil_to_4d(id2)

                print("testing identity", l)
                # image_ori = {
                #     "faces": faces if not two_ids else torch.cat([faces, face2], dim=-1),
                #     "ids": (torch.tensor([l], device=device)).unsqueeze(0).repeat(batch_size, 1) if not two_ids else torch.tensor([l, 7], device=device).unsqueeze(0).repeat(batch_size, 1),
                #     "num_ids": (torch.ones(batch_size, dtype=torch.long).to(device)) if not two_ids else torch.ones(batch_size, dtype=torch.long).to(device) * 2,
                #     "delta_w": torch.tensor(delta_w_dict[attr], device=device).repeat(batch_size, 1, 1) * weight_lambda if attr is not None else None,
                # }

                delta_w = None
                image_ori = {
                    "faces": faces if not two_ids else torch.cat(multi_faces_list, dim=-1),
                    "ids": (torch.tensor([l], device=device)).unsqueeze(0).repeat(batch_size, 1) if not two_ids else torch.tensor([l] * (len(multi_face)+1), device=device).unsqueeze(0).repeat(batch_size, 1),
                    "num_ids": (torch.ones(batch_size, dtype=torch.long).to(device)) if not two_ids else torch.ones(batch_size, dtype=torch.long).to(device) * (len(multi_face)+1),
                    "delta_w": delta_w,
                }

                precision_scope = nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            tic = time.time()
                            for n in trange(opt.n_iter, desc="Sampling"):
                                all_samples = list()
                                # prompts_idx = 0
                                # for prompts in tqdm(data, desc="data"):
                                if(opt.interpolate_ids[2] != -1):
                                    uc = None

                                    for seed in seed_list:
                                        opt.seed = seed
                                        seed_everything(seed)

                                        if opt.scale != 1.0:
                                            neg_prompt = temp_neg
                                            uc = model.get_learned_conditioning(batch_size * [neg_prompt],interpolate_ids=opt.interpolate_ids)
                                                                            
                                        prompt_mixing_prompt = [prompt.replace("sks person","brad pitt") for prompt in prompts]
                                        prompt_mixing_prompt = [prompt_mix.replace("ks person","ryan gosling") for prompt_mix in prompt_mixing_prompt]
                                        if(len(multi_face)>=2):
                                            prompt_mixing_prompt = [prompt_mix.replace("ata person","bill gates") for prompt_mix in prompt_mixing_prompt]
                                            prompt_mixing_prompt = [prompt_mix.replace("tre person","barack obama") for prompt_mix in prompt_mixing_prompt]
                                        print("prompt_mixing_prompt", prompt_mixing_prompt)
                                        image_for_ddim = {'face_img': image_ori["faces"], 'image_ori': image_ori, 'aligned_faces': None,
                                                        'caption': prompts, 'prompt_mixing_prompt': prompt_mixing_prompt, 
                                                        'steps_for_prompt_mixing': steps_for_prompt_mixing}

                                        if isinstance(prompts, tuple):
                                            prompts = list(prompts)

                                        
                                        uncond_prompt = [""]*batch_size
                                        # c = model.get_learned_conditioning(prompts, image_ori=image_ori,aligned_faces=None,interpolate_ids=opt.interpolate_ids)
                                        c = model.get_learned_conditioning(uncond_prompt)
                                        # c= model.cond_stage_model.encode(uncond_prompt, embedding_manager=model.embedding_manager,
                                        #                                 face_img=image_ori['faces'], image_ori=image_ori,
                                        #                                 aligned_faces=None,
                                        #                                 only_embedding=True)
                                        
                                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                        conditioning=c,
                                                                        batch_size=opt.n_samples,
                                                                        shape=shape,
                                                                        verbose=False,
                                                                        unconditional_guidance_scale=opt.scale,
                                                                        unconditional_conditioning=uc,
                                                                        eta=opt.ddim_eta,
                                                                        x_T=start_code,
                                                                        image_for_ddim=image_for_ddim,
                                                                        use_prompt_mixing=use_promt_mixing,)

                                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                                        if not opt.skip_save:
                                            for x_sample in x_samples_ddim:
                                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                                    os.path.join(sample_path, f"{base_count:05}.jpg"))
                                                base_count += 1

                                        if not opt.skip_grid:
                                            all_samples.append(x_samples_ddim)

                                        if not opt.skip_grid:
                                            # additionally, save as grid
                                            n_rows = 5
                                            # new_all_samples = []
                                            # for i in range(len(all_samples)):
                                            #     if(i%n_rows==0 and i!=0):
                                            #         new_all_samples.append(torch.zeros_like(all_samples[i]))
                                            #     new_all_samples.append(all_samples[i])
                                            # all_samples = new_all_samples
                                            # del new_all_samples
                                            
                                            grid = torch.stack(all_samples, 0)  # (n,b,c,h,w)
                                            # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                                            
                                            # with open(config.data.params.validation.params.pickle_path, "rb") as f:
                                            # data = pickle.load(f)
                                            input_img = np.array(id1)
                                            if(two_ids):
                                                input_img = cv2.resize(input_img, (grid.shape[-1], grid.shape[-1]))
                                                input_img = torch.from_numpy(input_img).to("cuda:0")/255
                                                input_img = input_img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                                                grid = torch.cat([input_img, grid], axis=1)
                        
                                            for idx, one_grid in enumerate(grid):
                                                if(idx==0):
                                                    one_grid = make_grid(one_grid, nrow=n_rows)
                                                    # to image
                                                    one_grid = 255. * rearrange(one_grid, 'c h w -> h w c').cpu().numpy()
                                                    # # Newly added for sinle img saving
                                                    # print("one_grid shape", grid.shape)
                                                    if(not two_ids):
                                                        grid = 255. * grid[0].permute(0,2,3,1).cpu().numpy()
                                                        for l in range(opt.n_samples):
                                                            Image.fromarray(grid[l].astype(np.uint8)).save(
                                                                os.path.join(outpath, f'{grid_count:04}-{prompt_all[prompts_idx].replace(" ", "-")}_'
                                                                                    f'{opt.img_suffix[:40]}_{l}.jpg'))
                                                    else:
                                                        Image.fromarray(one_grid.astype(np.uint8)).save(
                                                            os.path.join(outpath, f'{grid_count:04}-{prompt_all[prompts_idx].replace(" ", "-")}_'
                                                                                f'{opt.img_suffix[:40]}.jpg'))
                                                    grid_count += 1

                                            all_samples = list()
                                            del grid, one_grid, x_samples_ddim, samples_ddim, c, image_for_ddim, input_img, uc, uncond_prompt, prompt_mixing_prompt
                                    prompts_idx += 1
                                
                                else:
                                    for i in range(opt.interpolate_ids[3]+1):
                                        opt.interpolate_ids[2] = i
                                        uc = None
                                        if opt.scale != 1.0:
                                            neg_prompt = temp_neg
                                            uc = model.get_learned_conditioning(batch_size * [neg_prompt],interpolate_ids=opt.interpolate_ids)
                                        if isinstance(prompts, tuple):
                                            prompts = list(prompts)
                                        seed_everything(opt.seed)
                                        c = model.get_learned_conditioning(prompts, image_ori=image_ori,interpolate_ids=opt.interpolate_ids)
                                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                        conditioning=c,
                                                                        batch_size=opt.n_samples,
                                                                        shape=shape,
                                                                        verbose=False,
                                                                        unconditional_guidance_scale=opt.scale,
                                                                        unconditional_conditioning=uc,
                                                                        eta=opt.ddim_eta,
                                                                        x_T=start_code)

                                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                                        if not opt.skip_save:
                                            for x_sample in x_samples_ddim:
                                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                                    os.path.join(sample_path, f"{base_count:05}.jpg"))
                                                base_count += 1

                                        if not opt.skip_grid:
                                            all_samples.append(x_samples_ddim)

                                    if not opt.skip_grid:
                                        # additionally, save as grid
                                        grid = torch.stack(all_samples, 0)  # (n,b,c,h,w)
                                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                                        with open(config.data.params.validation.params.pickle_path, "rb") as f:
                                                data = pickle.load(f)
                                                if(len(data)==1):
                                                    input_img = cv2.imread(data[0])
                                                    input_img = cv2.resize(input_img, (grid.shape[-1], grid.shape[-1]))
                                                    input_img = torch.from_numpy(input_img).to("cuda:0")/255
                                                    input_img = input_img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                                                    one_grid = torch.cat([input_img, grid], axis=1)

                                        # for idx, one_grid in enumerate(grid):
                                        one_grid = make_grid(grid, nrow=n_rows)
                                        # to image
                                        one_grid = 255. * rearrange(one_grid, 'c h w -> h w c').cpu().numpy()

                                        Image.fromarray(one_grid.astype(np.uint8)).save(
                                            os.path.join(outpath, f'{grid_count:04}-{prompt_all[prompts_idx].replace(" ", "-")}_'
                                                                f'{opt.img_suffix[:40]}.jpg'))
                                        grid_count += 1

                                        all_samples = list()

                                        prompts_idx += 1

                    break

                            # toc = time.time()

            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
                f" \nEnjoy.")


if __name__ == "__main__":
    main()
