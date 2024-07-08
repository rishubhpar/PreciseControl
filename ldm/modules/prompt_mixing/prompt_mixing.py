import torch
from scipy.signal import medfilt2d
from torchvision import transforms

class PromptMixing:
    def __init__(self, args, object_of_interest_index, objects_to_preserve, avg_cross_attn=None,orig_mask=None):
        self.object_of_interest_index = object_of_interest_index
        # self.objects_to_preserve = [args.prompt.split(" ").index(o) + 1 for o in args.objects_to_preserve]
        self.objects_to_preserve = objects_to_preserve
        self.obj_pixels_injection_threshold = args.obj_pixels_injection_threshold

        self.start_other_prompt_range = args.start_prompt_range
        self.end_other_prompt_range = args.end_prompt_range

        self.start_cross_attn_replace_range = args.num_diffusion_steps
        self.end_cross_attn_replace_range = args.num_diffusion_steps

        self.start_self_attn_replace_range = 0
        self.end_self_attn_replace_range = args.end_preserved_obj_self_attn_masking
        self.remove_obj_from_self_mask = args.remove_obj_from_self_mask
        self.avg_cross_attn = avg_cross_attn
        self.orig_mask = orig_mask
        
        self.low_resource = args.low_resource

    def get_context_for_v(self, t, context, other_context):
        if other_context is not None and \
           self.start_other_prompt_range <= t < self.end_other_prompt_range:
            if self.low_resource:
                return other_context
            else:
                v_context = context.clone()
                # first half of context is for the uncoditioned image
                v_context[v_context.shape[0]//2:] = other_context
                return v_context
        else:
            return context

    def get_cross_attn(self, diffusion_model_wrapper, t, attn, place_in_unet, batch_size):
        if self.start_cross_attn_replace_range <= t < self.end_cross_attn_replace_range:
            if self.low_resource:
                for idx in self.object_of_interest_index:
                    attn[:, :, idx] = 0.2 * torch.from_numpy(medfilt2d(attn[:, :, idx].cpu().numpy(), kernel_size=3)).to(attn.device) + \
                                      0.8 * attn[:, :, idx]
                # attn[:,:,self.object_of_interest_index] = 0.2 * torch.from_numpy(medfilt2d(attn[:, :, self.object_of_interest_index].cpu().numpy(), kernel_size=3)).to(attn.device) + \
                #                                           0.8 * attn[:, :, self.object_of_interest_index]
            else:
                # first half of attn maps is for the uncoditioned image
                min_h = attn.shape[0] // 2
                for idx in self.object_of_interest_index:
                    attn[min_h:, :, idx] = 0.2 * torch.from_numpy(medfilt2d(attn[min_h:, :, idx].cpu().numpy(), kernel_size=3)).to(attn.device) + \
                                           0.8 * attn[min_h:, :, idx]
                
                    # attn[min_h:, :, idx] = attn[min_h:, :, idx] * transforms.Resize((attn.shape[1],attn.shape[2]))(self.orig_mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                # attn[min_h:, :, self.object_of_interest_index] = 0.2 * torch.from_numpy(medfilt2d(attn[min_h:, :, self.object_of_interest_index].cpu().numpy(), kernel_size=3)).to(attn.device) + \
                #                                                  0.8 * attn[min_h:, :, self.object_of_interest_index]
        return attn

    def get_self_attn(self, diffusion_model_wrapper, t, attn, place_in_unet, batch_size):
        if attn.shape[1] <= 32 ** 2 and \
           self.avg_cross_attn is not None and \
           self.start_self_attn_replace_range <= t < self.end_self_attn_replace_range:
            key = f"{place_in_unet}_cross"
            attn_index = getattr(diffusion_model_wrapper, f'{key}_index')
            cr = self.avg_cross_attn[key][attn_index]
            setattr(diffusion_model_wrapper, f'{key}_index', attn_index+1)

            if self.low_resource:
                attn = self.mask_self_attn_patches(attn, cr, batch_size)
            else:
                # first half of attn maps is for the uncoditioned image
                attn[attn.shape[0]//2:] = self.mask_self_attn_patches(attn[attn.shape[0]//2:], cr, batch_size//2)

        return attn
    
    def mask_self_attn_patches(self, self_attn, cross_attn, batch_size):
        h = self_attn.shape[0] // batch_size
        tokens = self.objects_to_preserve
        obj_token = self.object_of_interest_index

        normalized_cross_attn =  cross_attn - cross_attn.min()
        normalized_cross_attn /= normalized_cross_attn.max()

        mask = torch.zeros_like(self_attn[0])
        for tk in tokens:
            # import matplotlib.pyplot as plt
            # plt.imshow(normalized_cross_attn[:,:,tk].cpu().numpy())
            # plt.savefig(f"cross_attn_{tk}.png")
            mask_tk_in = torch.unique((normalized_cross_attn[:,:,tk] > self.obj_pixels_injection_threshold).nonzero(as_tuple=True)[1])
            mask[mask_tk_in, :] = 1
            mask[:, mask_tk_in] = 1

        if self.remove_obj_from_self_mask:
            for obj_tok in obj_token:
                obj_patches = torch.unique((normalized_cross_attn[:,:,obj_tok] > self.obj_pixels_injection_threshold).nonzero(as_tuple=True)[1])
                mask[obj_patches, :] = 0
                mask[:, obj_patches] = 0
            # obj_patches = torch.unique((normalized_cross_attn[:,:,obj_token] > self.obj_pixels_injection_threshold).nonzero(as_tuple=True)[1])
            # mask[obj_patches, :] = 0
            # mask[:, obj_patches] = 0
        # import matplotlib.pyplot as plt
        # plt.imshow(mask.cpu().numpy())
        # plt.savefig(f"mask_self_attn.png")
 
        self_attn[h:] = self_attn[h:] * (1 - mask) + self_attn[:h].repeat(batch_size - 1, 1, 1) * mask
        return self_attn