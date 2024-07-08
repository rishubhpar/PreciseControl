import torch
import numpy as np
import abc
from typing import Optional, Union, Tuple, Dict
import ldm.modules.prompt_mixing.seq_aligner as seq_aligner


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                # attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, low_resource):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource
        
 

class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class DummyController:
    def __call__(self, *args):
        return args[0]

    def __init__(self):
        self.num_att_layers = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"input_cross": [], "middle_cross": [], "output_cross": [],
                "input_self": [], "middle_self": [], "output_self": []}

    @staticmethod
    def get_empty_cross_store():
        return {"input_cross": [], "middle_cross": [], "output_cross": []}
    # def current_cross_attention(self, attn, is_cross: bool, place_in_unet: str):
    #     key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #     if key not in self.current_cross_attention:
    #         self.current_cross_attention[key] = attn
    
    def get_current_cross_attention(self):
        return self.current_cross_attention

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach())
            # if(is_cross):
            #     self.step_store_curr_cross[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
            # self.current_cross_attention = self.step_store_curr_cross
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
                    # if(key == "input_cross" or key == "middle_cross" or key == "output_cross"):
                    #     self.current_cross_attention[key][i] = self.step_store_curr_cross[key][i]

        self.step_store = self.get_empty_store()
        # self.step_store_curr_cross = self.get_empty_cross_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        # self.step_store_curr_cross = self.get_empty_cross_store()
        self.attention_store = {}
        # self.current_cross_attention = {}

    def __init__(self, low_resource,model_id="1"):
        super(AttentionStore, self).__init__(low_resource)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        # self.step_store_curr_cross = self.get_empty_cross_store()
        # self.current_cross_attention = {}
        self.model_id = model_id


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # print("attn shape in controller edit, and batch size", attn.shape, self.batch_size)
            attn_base, attn_repalce = attn[0], attn[1:]
            # print("attn_base shape, attn_replace shape", attn_base.shape, attn_repalce.shape)
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                # print("attn_repalce_new shape", attn_repalce_new.shape, alpha_words.shape, (self.replace_cross_attention(attn_base, attn_repalce) * alpha_words).shape)
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, tokenizer, device, low_resource, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(AttentionControlEdit, self).__init__(low_resource)
        # print("len of prompts", len(prompts['caption']*2))
        self.batch_size = len(prompts['caption'])*2
        self.tokenizer = tokenizer
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts["caption"]*2, num_steps, cross_replace_steps,
                                                                            self.tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper.to(attn_base.dtype))

    def __init__(self, prompts, tokenizer, device, low_resource, num_steps: int, cross_replace_steps: float, self_replace_steps: float):
        super(AttentionReplace, self).__init__(prompts, tokenizer, device, low_resource, num_steps, cross_replace_steps, self_replace_steps)

        # self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to(device)
        self.mapper = torch.eye(77, 77).to(device).unsqueeze(0)


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words