"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from cv2 import dilate
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from ldm.modules.prompt_mixing.attention_based_segmentation2 import Segmentor
from ldm.modules.prompt_mixing.attention_utils import show_cross_attention, aggregate_attention, get_current_cross_attn
# from ldm.modules.prompt_mixing.prompt_to_prompt_controllers_astar import DummyController, AttentionStore
from ldm.modules.prompt_mixing.prompt_to_prompt_controllers import AttentionStore ,DummyController

from src.grounded_sam.grounded_sam_demo import main as get_sam_mask

import dlib
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import cv2
from pytorch_lightning import seed_everything
from torchvision import transforms
import torchvision
def get_bbox(dets, shape_ratio, cv2=False):
    if(not cv2):
        tl_x = int(dets.left() * shape_ratio)
        tl_y = int(dets.top() * shape_ratio)
        br_x = int(dets.right() * shape_ratio)
        br_y = int(dets.bottom() * shape_ratio)
        w = br_x - tl_x
        h = br_y - tl_y
    else:
        tl_x, tl_y, w, h = int(dets[0] * shape_ratio), int(dets[1] * shape_ratio), int(dets[2] * shape_ratio), int(dets[3] * shape_ratio)
        # tl_x = min(0, int(dets[0] * shape_ratio)-1)
        # tl_y = min(0, int(dets[1] * shape_ratio)-1)
        # w = max(int(dets[2] * shape_ratio)+2, 32)
        # h = max(int(dets[3] * shape_ratio)+2,32)
    return (tl_x, tl_y, w, h)

def get_face_bounding_boxes(face_img):
    import json
    shape_ratio = 64 / face_img.shape[0]
    # mask_json = json.load(open("./mask_beach.json"))
    mask_json = get_sam_mask(Image.fromarray(face_img[:,:,::-1]))
    mask1 = torch.from_numpy(np.array(mask_json[1]["mask"]))
    mask2 = torch.from_numpy(np.array(mask_json[2]["mask"]))
    bg_masks = torch.bitwise_not(torch.bitwise_or(mask1, mask2))
    mask1 = transforms.Resize((64,64))(mask1.float()).view(1,1,64,64)
    mask2 = transforms.Resize((64,64))(mask2.float()).view(1,1,64,64)
    bg_masks = transforms.Resize((64,64))(bg_masks.float()).view(1,1,64,64)
    plt.imshow(np.concatenate([mask1.numpy()[0,0], mask2.numpy()[0,0], bg_masks.numpy()[0,0]], axis=1))
    plt.savefig("./pz_masks.png")
    # masks = torch.cat([bg_masks, mask1, mask2], dim=0).numpy()
    masks = [bg_masks, mask1, mask2]

    # print("masks shape : ", masks.shape)
    bboxs = []
    x1, y1, x2, y2 = mask_json[1]["box"]
    bboxs.append(get_bbox((x1, y1, x2, y2), shape_ratio, cv2=True))
    x1, y1, x2, y2 = mask_json[2]["box"]
    bboxs.append(get_bbox((x1, y1, x2, y2), shape_ratio, cv2=True))
    # detector = dlib.get_frontal_face_detector()
    # dets = detector(face_img, 1)
    # bboxs = [get_bbox(det, shape_ratio) for det in dets]
    face_img = cv2.UMat(face_img).get()
    face_img = cv2.resize(face_img, (64,64))
    for b, (x, y , w, h) in enumerate(bboxs):
        color = (255, 0, 0) if b == 0 else (0, 255, 0)
        cv2.rectangle(face_img, (x,y), (x+w, y+h), color,1)
    plt.imshow(face_img[:,:,::-1])
    plt.savefig("./pz_bbox_on_face.png")
    # print("prediceted bounding boxes : ", bboxs)
    return bboxs, masks

def assign_bbox_to_ids(model, model_config, args, **kwargs):
    seed_everything(kwargs["image_for_ddim"]["seed"])
    controller = AttentionStore(args.low_resource)
    ddim_sampler = DDIMSamplerWrapper(model=model, controller=controller, model_config=model_config)
    image, x_t, orig_all_latents, _ = ddim_sampler.sample(args, do_multi_diff=False, **kwargs)
    torchvision.utils.save_image(transforms.ToTensor()(image[0]), "./pmm_sample_img0.jpg")
    bboxs, masks = get_face_bounding_boxes(image[0][:,:,::-1].astype(np.uint8))
    idx1 = kwargs["image_for_ddim"]["caption"][-1].split(" ").index("sks") + 2
    idx2 = kwargs["image_for_ddim"]["caption"][-1].split(" ").index("ks") + 1
    attn = aggregate_attention(controller, res=16, from_where=("output", "input"), prompts=kwargs["image_for_ddim"]['caption'],
                                                    is_cross=True, select=len(kwargs["image_for_ddim"]['caption']) - 1)
    token_attn = attn[:,:,idx1]
    # print("grad is available for variable : ", token_attn.requires_grad)
    # do token_attn.repeat(2, axis=0).repeat(2, axis=1) similar operation in torch
    curr_noun_map = token_attn.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    normalised_noun_map1 = torch.nn.functional.normalize(curr_noun_map, p=2, dim=(0,1))
    # mask_A = normalised_noun_map1.detach() > 0.04
    plt.imshow(normalised_noun_map1.detach().cpu().numpy())
    plt.savefig(os.path.join("pz_final_cross_attn1_{}.png".format(str(50).zfill(2))))

    token_attn = attn[:,:,idx2]
    curr_noun_map = token_attn.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    normalised_noun_map2 = torch.nn.functional.normalize(curr_noun_map, p=2, dim=(0,1))
    # mask_B = normalised_noun_map2.detach() > 0.04
    plt.imshow(normalised_noun_map2.detach().cpu().numpy())
    plt.savefig(os.path.join("pz_final_cross_attn2_{}.png".format(str(50).zfill(2))))

    # matching bbox with cross attn map
    x1, y1, w1, h1 = int(np.floor(bboxs[0][0]/2)), int(np.floor(bboxs[0][1]/2)), int(np.ceil(bboxs[0][2]/2)), int(np.ceil(bboxs[0][3]/2))
    x2, y2, w2, h2 = int(np.floor(bboxs[1][0]/2)), int(np.floor(bboxs[1][1]/2)), int(np.ceil(bboxs[1][2]/2)), int(np.ceil(bboxs[1][3]/2))
    id1_iou1 = torch.sum(normalised_noun_map1[y1:y1+h1, x1:x1+w1])
    id1_iou2 = torch.sum(normalised_noun_map1[y2:y2+h2, x2:x2+w2])

    # id2_iou1 = torch.sum(mask_A[bboxs[0][0]:bboxs[0][2], bboxs[0][1]:bboxs[0][3]])
    # id2_iou2 = torch.sum(mask_A[bboxs[1][0]:bboxs[1][2], bboxs[1][1]:bboxs[1][3]])

    matched_bbox = {}
    if(id1_iou1 > id1_iou2):
        final_mask = torch.cat([masks[0], masks[1], masks[2]], dim=0).numpy()
    else:
        # print("[mask swapped]")
        final_mask = torch.cat([masks[0], masks[2], masks[1]], dim=0).numpy()
    
    # final_maskA = torch.zeros_like(mask_A)
    # final_maskB = torch.zeros_like(mask_B)
    # box = matched_bbox["id1"][0]
    # final_maskA[box[0]:box[0]+box[2],box[1]:box[1]+box[3]] = 1
    # box = matched_bbox["id2"][0]
    # final_maskB[box[0]:box[0]+box[2],box[1]:box[1]+box[3]] = 1

    # plt.imshow(np.fliplr(np.rot90(final_maskA.detach().cpu().numpy(),3)))
    # plt.savefig(os.path.join("pz_maska_{}.png".format(str(50).zfill(2))))
    # plt.imshow(np.fliplr(np.rot90(final_maskB.detach().cpu().numpy(),3)))
    # plt.savefig(os.path.join("pz_maskb_{}.png".format(str(50).zfill(2))))

    del controller, ddim_sampler

    return final_mask, image


def generate_original_image(model, model_config, args, **kwargs):

    with torch.no_grad():
        masks, image = assign_bbox_to_ids(model, model_config, args, **kwargs)
    seed_everything(kwargs["image_for_ddim"]["seed"])
    kwargs["image_for_ddim"]["masks"] = masks
    controller = AttentionStore(args.low_resource)
    ddim_sampler = DDIMSamplerWrapper(model=model, controller=controller, model_config=model_config)
    # image, x_t, orig_all_latents, _ = ddim_sampler.sample(args, do_multi_diff=False, **kwargs)
    # orig_mask = Segmentor(controller, kwargs["image_for_ddim"]['caption'], args.num_segments, args.background_segment_threshold, 
    #                       background_nouns=args.background_nouns).get_background_mask(kwargs["image_for_ddim"]["caption"][-1].split(" ").index("sks")+1)
    # average_attention = controller.get_average_attention()
    x_t, orig_all_latents = None, None
    orig_mask = None
    average_attention = None
    return image, x_t, orig_all_latents, orig_mask, average_attention, controller


class DDIMSamplerWrapper(object):
    def __init__(self, model, schedule="linear", controller=None, prompt_mixing=None, model_config=None,**kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.controller = controller
        if self.controller is None:
            self.controller = DummyController()
        self.prompt_mixing = prompt_mixing
        self.model_config = model_config
        self.enbale_attn_controller_changes = False
        self.register_attention_control()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               args,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               image_for_ddim=None,
               orig_image_for_ddim=None,
               multi_diff_ddim=None,
               use_prompt_mixing=False,
               do_multi_diff = True,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        image, _ , all_latents, object_mask = self.ddim_sampling(args, conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    image_for_ddim=image_for_ddim,
                                                    orig_image_for_ddim=orig_image_for_ddim,
                                                    multi_diff_ddim=multi_diff_ddim,
                                                    use_prompt_mixing=use_prompt_mixing,
                                                    do_multi_diff = do_multi_diff,
                                                    **kwargs)
        return image, x_T, all_latents, object_mask

    @torch.no_grad()
    def ddim_sampling(self, args, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,image_for_ddim=None,orig_image_for_ddim=None, use_prompt_mixing=False,
                      post_background = False, orig_all_latents = None, orig_mask = None, do_multi_diff=False, multi_diff_ddim=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            # img = torch.randn(shape, device=device)
            img = torch.randn((4,shape[1],shape[2],shape[3]), device=device)
            img = img[image_for_ddim["sample_id"]].unsqueeze(0)
            # if(orig_image_for_ddim is not None):
            #     img = torch.cat([img,img])
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, position=0)

        # multi diffusion
        num_of_prompts = 3
        background_color = torch.rand(num_of_prompts, 3, device=device)[:, :, None, None].repeat(1, 1, 512, 512)
        bootstrap_background = torch.cat([self.model.get_first_stage_encoding(self.model.encode_first_stage(bg.unsqueeze(0))) for bg in background_color])
        noise = img.clone().repeat(num_of_prompts - 1, 1, 1, 1)
        views = image_for_ddim.get("masks", None)
        count = torch.zeros_like(img)
        value = torch.zeros_like(img)

        self.enbale_attn_controller_changes = True if multi_diff_ddim is None else False
        object_mask = None
        self.diff_step = 0
        all_latents = []
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            self.input_cross_index = 0
            self.middle_cross_index = 0
            self.output_cross_index = 0
            # getting condition from mapper
            # TODO: Don't hardcode c and checkwhether passing img to unet is correct (removed hardcoding, passing im is wrong, img is latent img)
            count.zero_()
            value.zero_()


            if image_for_ddim is not None:
                two_ids = image_for_ddim.get('two_ids', False)
                face_img = image_for_ddim['face_img']
                img_ori = image_for_ddim['image_ori']
                aligned_faces = image_for_ddim['aligned_faces']
                c = image_for_ddim['caption']
                masks = image_for_ddim.get("masks",None)
                if(masks is not None):
                    masks = image_for_ddim["masks"]
                    # mask_prompt_list = [" photo of kitchen", "a photo of sks person baking a cake", "a photo of ks person baking a cake"]
                    if(multi_diff_ddim is not None):
                        md_face_img = multi_diff_ddim['face_img']
                        md_img_ori = multi_diff_ddim['image_ori']
                        mask_prompt_list = md_img_ori["multiple_prompt"]
                        if(md_img_ori.get("model2") is not None):
                            id1_dict = md_img_ori["id1_multi_diff"]
                            id2_dict = md_img_ori["id2_multi_diff"]
                            h_space_id1 = {'h_space_feat': None, 't':ts.repeat(1)}
                            h_space_id2 = {'h_space_feat': None, 't':ts.repeat(1)}
                if(use_prompt_mixing):
                    steps_for_prompt_mixing = image_for_ddim['steps_for_prompt_mixing']
            # h_space_feature = self.model.apply_model(img, ts, (uc,None), return_hspace=True)
            # h_space = {'h_space_feat': h_space_feature, 't':ts}
            h_space = {'h_space_feat': None, 't':ts}
            
            other_cond = None
            if(not do_multi_diff):
                if(use_prompt_mixing and i < steps_for_prompt_mixing):
                    prompt_mixing_text = image_for_ddim['prompt_mixing_prompt']
                    # print("prompt mixing")
                    c = prompt_mixing_text
                    cond = self.model.get_learned_conditioning(c, face_img=face_img, image_ori=img_ori,aligned_faces=aligned_faces,h_space=h_space)
                    # getting other context
                    # if(orig_image_for_ddim is not None):
                    #     other_cond = self.model.get_learned_conditioning(c, face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                    #                                                      h_space=h_space)
                    
                else:
                    # c = c*face_img.shape[0]
                    cond = self.model.get_learned_conditioning(c, face_img=face_img, image_ori=img_ori,aligned_faces=aligned_faces,h_space=h_space)
                    
                # getting other context
                if(orig_image_for_ddim is not None):
                    other_cond = self.model.get_learned_conditioning(c, face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                                                                        h_space=h_space)
                    other_cond = torch.cat([cond, other_cond])

                outs = self.p_sample_ddim(args, img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning, other_cond=other_cond, orig_image_for_ddim=None)
                
                # cond_list_for_each_timestep.append(cond.detach().cpu().numpy())
                img = outs[0]
                img = self.controller.step_callback(img)
            else:
                # for idx, view in enumerate(views):
                masks = torch.from_numpy(views).float().cuda()
                
                latent_view = img.clone().repeat(num_of_prompts, 1, 1, 1)
                mask_prompt = mask_prompt_list

                # if(md_img_ori.get("model2") is not None):
                #     latent_view_id2 = latent_view[2:]
                #     latent_view = latent_view[:2]
                #     masks_id2 = masks[2:]
                #     masks = masks[:2]
                #     mask_prompt_id2 = mask_prompt[2:]
                #     mask_prompt = mask_prompt[:2]

                if(i < image_for_ddim["bootstrapping_steps"]):
                    bg = bootstrap_background[torch.randint(0, num_of_prompts, (num_of_prompts-1,))]
                    bg = self.model.q_sample(bg, t=ts, noise=noise)
                    latent_view[1:] = latent_view[1:] * masks[1:] + (1 - masks[1:]) * bg


                if(use_prompt_mixing and i < steps_for_prompt_mixing):
                    prompt_mixing_text = image_for_ddim['prompt_mixing_prompt']
                    # print("prompt mixing")
                    c = prompt_mixing_text
                    cond = self.model.get_learned_conditioning(c, face_img=face_img, image_ori=img_ori,aligned_faces=aligned_faces,h_space=h_space)

                    outs = self.p_sample_ddim(args, img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning, other_cond=other_cond, orig_image_for_ddim=None)

                    img = outs[0]
                    # getting other context
                    # if(orig_image_for_ddim is not None):
                    #     other_cond = self.model.get_learned_conditioning(c, face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                    #                                                      h_space=h_space)
                    
                else:
                    # c = c*face_img.shape[0]
                    if(h_space["t"].shape[0]!=len(mask_prompt)):
                            h_space["t"] = h_space["t"].repeat(len(mask_prompt))

                    if(md_img_ori.get("model2") is None):
                        # only image_ori and ts are use so change that variable
                        cond = self.model.get_learned_conditioning(mask_prompt, face_img=md_face_img, image_ori=md_img_ori,aligned_faces=aligned_faces,h_space=h_space)
                        

                        # getting other context
                        if(orig_image_for_ddim is not None):
                            other_cond = self.model.get_learned_conditioning(mask_prompt, md_face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                                                                                h_space=h_space)
                            other_cond = torch.cat([cond, other_cond])

                        outs = self.p_sample_ddim(args, latent_view, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                quantize_denoised=quantize_denoised, temperature=temperature,
                                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning, other_cond=other_cond, orig_image_for_ddim=None)

                        # bootstrap_background[idx] = outs[0][1] * (1 - masks)
                        # update value where mask is 1
                        value += (outs[0] * masks).sum(dim=0,keepdims=True)
                        count += masks.sum(dim=0, keepdims=True)

                        img = torch.where(count > 0, value / count, value)
                    
                    else:
                        cond_id0 = md_img_ori.get("model2")[0].model.get_learned_conditioning(mask_prompt[:1], face_img=md_face_img, image_ori=orig_image_for_ddim['image_ori'],
                                                                       aligned_faces=aligned_faces,h_space=h_space_id1)
                        cond_id1 = self.model.get_learned_conditioning(mask_prompt[1:2], face_img=md_face_img, image_ori=orig_image_for_ddim['image_ori'],
                                                                       aligned_faces=aligned_faces,h_space=h_space_id1)
                        cond_id2 = md_img_ori.get("model2")[1].model.get_learned_conditioning(mask_prompt[2:], face_img=md_face_img, image_ori=orig_image_for_ddim['image_ori'],
                                                                                     aligned_faces=aligned_faces,h_space=h_space_id2)
                        cond = torch.cat([cond_id0, cond_id1, cond_id2])
                        # getting other context
                        if(orig_image_for_ddim is not None):
                            other_cond0 = md_img_ori.get("model2")[0].model.get_learned_conditioning(mask_prompt[:1], md_face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                                                                                h_space=h_space_id1)
                            other_cond1 = self.model.get_learned_conditioning(mask_prompt[1:2], md_face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                                                                                h_space=h_space_id1)
                            other_cond2 = md_img_ori.get("model2")[1].model.get_learned_conditioning(mask_prompt[2:], md_face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                                                                                h_space=h_space_id2)
                            other_cond = torch.cat([other_cond0, other_cond1, other_cond2])
                            other_cond = torch.cat([cond, other_cond])

                        outs_id0 = self.p_sample_ddim(args, latent_view[:1], cond[:1], ts, index=index, use_original_steps=ddim_use_original_steps,
                                                quantize_denoised=quantize_denoised, temperature=temperature,
                                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning, other_cond=other_cond0, orig_image_for_ddim=None,
                                                model2=md_img_ori.get("model2")[0],)
                        outs_id1 = self.p_sample_ddim(args, latent_view[1:2], cond[1:2], ts, index=index, use_original_steps=ddim_use_original_steps,
                                                quantize_denoised=quantize_denoised, temperature=temperature,
                                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning, other_cond=other_cond1, orig_image_for_ddim=None)
                        outs_id2 = self.p_sample_ddim(args, latent_view[2:], cond[2:], ts, index=index, use_original_steps=ddim_use_original_steps,
                                                quantize_denoised=quantize_denoised, temperature=temperature,
                                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning, other_cond=other_cond2, orig_image_for_ddim=None,
                                                model2=md_img_ori.get("model2")[1],
                                                )
                        
                        value_id0 = (outs_id0[0] * masks[:1]).sum(dim=0,keepdims=True)
                        count_id0 = masks[:1].sum(dim=0, keepdims=True)
                        value_id1 = (outs_id1[0] * masks[1:2]).sum(dim=0,keepdims=True)
                        count_id1 = masks[1:2].sum(dim=0, keepdims=True)
                        value_id2 = (outs_id2[0] * masks[2:]).sum(dim=0,keepdims=True)
                        count_id2 = masks[2:].sum(dim=0, keepdims=True)
                        
                        value = value_id0 + value_id1 + value_id2
                        count = count_id0 + count_id1 + count_id2
                        img = torch.where(count > 0, value / count, value)
                # cond_list_for_each_timestep.append(cond.detach().cpu().numpy())
                # img = outs[0]
                # img = self.controller.step_callback(img)

                for param in self.model.model.diffusion_model.parameters():
                    param.grad = None
                for param in self.model.e4e_encoder.parameters():
                    param.grad = None

                torch.cuda.empty_cache()
            # img = outs[0]

            pm_and_matching_args = {}
            object_mask = None
            prompt = image_for_ddim['caption']
            
            if post_background and (self.diff_step == args.background_blend_timestep):
                object_mask = Segmentor(self.cross_attn_ioubackground_segment_threshold,
                                        background_nouns=args.background_nouns)\
                    .get_background_mask(orig_image_for_ddim["caption"][-1].split(" ").index("sks")+1)
                self.enbale_attn_controller_changes = False
                pm_mask = object_mask.astype(np.bool8) + orig_mask.astype(np.bool8)
                pm_mask = torch.from_numpy(pm_mask).float().cuda()
                shape = (1, 1, pm_mask.shape[0], pm_mask.shape[1])
                pm_mask = torch.nn.Upsample(size=(64, 64), mode='nearest')(pm_mask.view(shape))
                # plt.imshow(pm_mask.cpu().numpy()[0, 0])
                # plt.savefig(f"./pm_masks/pm_mask_{i}.png")
                mask_eroded = dilate(pm_mask.cpu().numpy()[0, 0], np.ones((3, 3), np.uint8), iterations=1)
                pm_mask = torch.from_numpy(mask_eroded).float().cuda().view(1, 1, 64, 64)
                img = pm_mask * img + (1 - pm_mask) * orig_all_latents[self.diff_step]

            all_latents = []
            all_latents.append(img.detach())
            self.diff_step += 1

            # img, pred_x0 = outs

            # if(two_ids):
            #     # img = img - iou_alpha * transforms.Resize((64,64))(torch.from_numpy(grad_attn_iou).float().to(img.device).view(1,1,32,32))

            # if callback: callback(i)
            # if img_callback: img_callback(pred_x0, i)

            # if index % log_every_t == 0 or index == total_steps - 1:
            #     intermediates['x_inter'].append(img)
            #     intermediates['pred_x0'].append(pred_x0)

            del h_space, face_img, img_ori, aligned_faces, ts, c, cond, pm_and_matching_args
            if(two_ids and do_multi_diff):
                # del attn, token_attn, curr_noun_map, normalised_noun_map1, normalised_noun_map2
                pass
    
        # img = outs[0]

        image = self.latent2image(all_latents[-1])

        return image, None, all_latents, object_mask
    
    @torch.no_grad()
    def register_attention_control(self):
        def ca_forward(model_self, place_in_unet):
            to_out = model_self.to_out
            if type(to_out) is torch.nn.modules.container.ModuleList:
                to_out = model_self.to_out[0]
            else:
                to_out = model_self.to_out

            def forward(x, context=None, mask=None):
                batch_size, sequence_length, dim = x.shape
                h = model_self.heads
                q = model_self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else (x, None)

                k = model_self.to_k(context[0])
                if is_cross and self.prompt_mixing is not None:
                    v_context = self.prompt_mixing.get_context_for_v(self.diff_step, context[0], context[1])
                    v = model_self.to_v(v_context)
                else:
                    v = model_self.to_v(context[0])

                q = rearrange(q, "b t (h d) -> (b h) t d", h=h)
                k = rearrange(k, "b i (h d) -> (b h) i d", h=h)
                v = rearrange(v, "b i (h d) -> (b h) i d", h=h)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * model_self.scale

                if mask is not None:
                    mask = mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~mask, max_neg_value)

                # attention, what we cannot get enough of
                attn = sim.softmax(dim=-1)
                # print("place in unet : ", place_in_unet)
                # print("attn shape : ", attn.shape)
                if self.enbale_attn_controller_changes:
                    attn = self.controller(attn, is_cross, place_in_unet)

                
                # print("attn shape after self controller: ", attn.shape)
                if is_cross and self.prompt_mixing is not None and context[1] is not None:
                    attn = self.prompt_mixing.get_cross_attn(self, self.diff_step, attn, place_in_unet, batch_size)
                    

                if not is_cross and (not self.model_config["low_resource"] or not self.uncond_pred) and self.prompt_mixing is not None:
                    # print("self attn shape : ", attn.shape)
                    attn = self.prompt_mixing.get_self_attn(self, self.diff_step, attn, place_in_unet, batch_size)

                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = rearrange(out, "(b h) t d -> b t (h d)", h=h)
                return to_out(out)

            return forward

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.model.model.named_children().__iter__().__next__()[1].named_children()
        for net in sub_nets:
            # print(net[0])
            if "input" in net[0]:
                cross_att_count += register_recr(net[1], 0, "input")
            elif "output" in net[0]:
                cross_att_count += register_recr(net[1], 0, "output")
            elif "middle" in net[0]:
                cross_att_count += register_recr(net[1], 0, "middle")
        self.controller.num_att_layers = cross_att_count

    # @torch.no_grad()
    # def diffusion_step(self, latents, context, t, other_context=None):
    #     latents = self.p_sample_ddim(latents, context, t)[0]
    #     latents = self.controller.step_callback(latents)
    #     return latents
    
    @torch.no_grad()
    def latent2image(self, latents):
        x_samples_ddim = self.model.decode_first_stage(latents)
        image = (x_samples_ddim / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def p_sample_ddim(self, args, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, other_cond=None, orig_image_for_ddim=None, model2=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            self.uncond_pred = True
            c = (c, None)
            e_t = self.model.apply_model(x, t, c)
        else:
            n = x.shape[0] * 2 if orig_image_for_ddim is None else 4
            self.uncond_pred = False
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * n)
            if(n==4 and orig_image_for_ddim is not None):
                c_in = torch.cat([unconditional_conditioning, unconditional_conditioning, c, c])
            else:
                c_in = torch.cat([unconditional_conditioning.repeat(c.shape[0],1, 1), c])
            # if(other_cond is not None):
            #     other_cond = torch.cat([unconditional_conditioning, other_cond])
            # print("c_in shape, t_in shape :", c_in.shape, t_in.shape)
            c_in = (c_in, other_cond)
            # print("current memory used 3:", torch.cuda.memory_allocated())
            if(model2 is not None):
                e_t_uncond, e_t = model2.model.apply_model(x_in, t_in, c_in).chunk(2)
            else:
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            # print("current memory used 3.1:", torch.cuda.memory_allocated())
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        del a_t, a_prev, sigma_t, sqrt_one_minus_at, dir_xt, noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def init_latent(self, latent, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.model.in_channels, self.height // 8, self.width // 8),
                generator=self.generator, device=self.device
            )
        latents = latent.expand(batch_size,  self.model.in_channels, self.height // 8, self.width // 8).to(self.device)
        return latent, latents


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
    
    # def get_original_image(self, args,
    #            S,
    #            batch_size,
    #            shape,
    #            conditioning=None,
    #            callback=None,
    #            normals_sequence=None,
    #            img_callback=None,
    #            quantize_x0=False,
    #            eta=0.,
    #            mask=None,
    #            x0=None,
    #            temperature=1.,
    #            noise_dropout=0.,
    #            score_corrector=None,
    #            corrector_kwargs=None,
    #            verbose=True,
    #            x_T=None,
    #            log_every_t=100,
    #            unconditional_guidance_scale=1.,
    #            unconditional_conditioning=None,
    #            image_for_ddim=None,
    #            orig_image_for_ddim=None,
    #            use_prompt_mixing=False,
    #            # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    #            **kwargs):
    #     # controller = AttentionStore(args.ldm_stable_config["low_resource"])
    #     self.controller = AttentionStore(True)
    #     image, x_T, all_latents, _ = self.sample(args, S, batch_size, shape, conditioning, callback, normals_sequence,
    #                                              img_callback, quantize_x0, eta, mask, x0, temperature, noise_dropout,
    #                                              score_corrector, corrector_kwargs, verbose, x_T, log_every_t,
    #                                              unconditional_guidance_scale, unconditional_conditioning,
    #                                              image_for_ddim, orig_image_for_ddim, use_prompt_mixing, **kwargs)
    #     print("attention store keys :", self.controller.step_store.keys())
    #     print("attention store keys :", self.controller.attention_store.keys())
    #     orig_mask = Segmentor(self.controller, orig_image_for_ddim['caption'], args.num_segments, args.background_segment_threshold, 
    #                           background_nouns=args.background_nouns).get_background_mask("face")
    #     average_attention = self.controller.get_average_attention()
    #     return image, x_T, all_latents, orig_mask, average_attention