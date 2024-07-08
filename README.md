# Inserting Anybody in Diffusion Models via Celeb Basis

<a href='https://arxiv.org/abs/2306.00926'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://celeb-basis.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 


<div>
<span class="author-block">
<a href="https://scholar.google.com/citations?user=RaRoJFYAAAAJ&hl=en" target="_blank">Rishubh Parihar</a><sup>1,2</sup></span>,
<span class="author-block">
  <a href="http://vinthony.github.io/" target="_blank">Xiaodong Cun</a><sup>2</sup></span>,
<span class="author-block">
    <a href="https://yzhang2016.github.io" target="_blank">Yong Zhang</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=ym_t6QYAAAAJ&hl=zh-CN&oi=sra" target="_blank">Maomao Li</a><sup>2,*</sup>,
  </span>
<span class="author-block"><a href="https://chenyangqiqi.github.io/" target="_blank">Chenyang Qi</a><sup>3,2</sup></span>, <br>
  <span class="author-block">
    <a href="https://xinntao.github.io/" target="_blank">Xintao Wang</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ" target="_blank">Ying Shan</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CCUQi50AAAAJ" target="_blank">Huicheng Zheng</a><sup>1,*</sup>
  </span> (* Corresponding Authors)
  </div>

  
<div class="is-size-5 publication-authors">
                  <span class="author-block">
                  <sup>1</sup> Sun Yat-sen University &nbsp;&nbsp;&nbsp;
                  <sup>2</sup> Tencent AI Lab &nbsp;&nbsp;&nbsp;
                  <sup>3</sup> HKUST </span>
                </div>
<br>

**TL;DR: Intergrating a unique individual into the pre-trained diffusion model with:** 

✅ just <b>one</b> facial photograph &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
✅ in <b>3</b> minutes tunning &nbsp;&nbsp;&nbsp;&nbsp;  ✅ Genearte and interact with other (new person) concepts &nbsp;&nbsp;&nbsp;&nbsp;
✅ Do facial edits for generated images using our mapper neteork </br>

![Fig1](./assets/teaser-fig-precisecontrol.png)


### Updates
- **2024/07/15:** Code released!

### How It Work
<!-- ![Fig2](https://github.com/ygtxr1997/CelebBasis/assets/4397546/efe0eb13-0c74-45f0-9252-a49976dd228d)



First, we collect about 1,500 celebrity names as the initial collection. Then, we manually filter the initial one to m = 691 names, based on the synthesis quality of text-to-image diffusion model(stable-diffusion} with corresponding name prompt. Later, each filtered name is tokenized and encoded into a celeb embedding group. Finally, we conduct Principle Component Analysis to build a compact orthogonal basis.

![Fig4](https://github.com/ygtxr1997/CelebBasis/assets/4397546/fe70c970-f9d4-4255-bb76-0c6154778b4e)

We then personalize the model using input photo. During training~(left), we optimize the coefficients of the celeb basis with the help of a fixed face encoder. During inference~(right), we combine the learned personalized weights and shared celeb basis to generate images with the input identity.

More details can be found in our [project page](https://celeb-basis.github.io).
 -->

### Setup

Our code mainly bases on [CelebBasis](https://github.com/ygtxr1997/CelebBasis).
It also uses [Prompt-Mixing](https://github.com/orpatashnik/local-prompt-mixing) for background preservation, [Lora](https://github.com/cloneofsimo/lora) for lora finetuning and [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) for mask generation.
To set up our environment, please run:

```shell
conda env create -f environment.yaml
conda activate sd
```

And follow [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) for setting up mask prediction in two person generation.
The pre-trained weights used in this repo include [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) and 
[CosFace R100 trained on Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo).
You may copy these pre-trained weights to `./weights`, and the directory tree will be like:

```shell
CelebBasis/
  |-- weights/
      |--glint360k_cosface_r100_fp16_0.1/
          |-- backbone.pth (249MB)
      |--sv2-1_768-ema-pruned.ckpt (~7.0GB)
```

We use [PIPNet](https://github.com/jhb86253817/PIPNet) to align and crop the face.
The PIPNet pre-trained weights can be downloaded from [this link](https://github.com/ygtxr1997/CelebBasis/issues/2#issuecomment-1607775140) (provided by @justindujardin)
or our [Baidu Yun Drive](https://pan.baidu.com/s/1Cgw0i723SyeLo5lbJu-b0Q) with extracting code: `ygss`.
Please copy `epoch59.pth` and `FaceBoxesV2.pth` to `CelebBasis/evaluation/face_align/PIPNet/weights/`.

### Usage

#### 0. Face Alignment

To make the Face Recognition model work as expected, 
given an image of a person, 
we first align and crop the face following [FFHQ-Dataset](https://github.com/NVlabs/ffhq-dataset).

Assuming your image folder is `./aug_images/comparision` and the output folder is `./aug_images/comparision/edited/`,
you may run the following command to align & crop images.

```shell
bash ./00_align_face.sh ./aug_images/comparision ./aug_images/comparision/edited/
```

For example, we provide some cropped faces in `./aug_images/comparision/edited`

#### 1. Personalization

The training config file is `./configs/stable-diffusion/aigc_id_for_lora.yaml`.
The most important settings are listed as follows.

**Important Data Settings**
```yaml
data:
  params:
    batch_size: 2  # We use batch_size 2
    train:
      target: ldm.data.face_id.FFhq_dataset 
      params:
        root_dir: "./aug_images/lora_finetune_comparision_data/bengio/"
        split: train
        use_aug: False
        image_size: 512
        limit_dataset_size: -1
        use_data_interpolation: False
        percentage_of_synthetic_data: 0.1
        lora_finetuning: True
        multiple_samples: True
    validation:
      target: ldm.data.face_id.FFhq_dataset
      params:
        pickle_path: /Your/Path/To/Images/ffhq.pickle  # consistent with train.params.pickle_path
```

**Important Training Settings**
```yaml
lightning:
  modelcheckpoint:
    params:
      every_n_train_steps: 20
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 50
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
    max_steps: 50
    accumulate_grad_batches: 8
```

Reduce the accumulate grad batches as per the GPU availablity, but for lower value increase the max_steps appropriately.

**Training**
```shell
bash ./01_start_lora_finetuning.sh ./weights/v2-1_768-ema-pruned.ckpt
```

Consequently, a project folder named `id_name` is generated under `./logs`. 

#### 2. Generation

Edit the prompt file `./infer_images/example_prompt.txt`, where `sks` denotes the first identity. `image_name.jpg` should be present inside 
`./aug_images/comparision/edited/` or else manually you have to change the root dir in code.

<!-- Optionally, in `./02_start_test.sh`, you may modify the following var as you need:
```shell
step_list=(799)  # the step of trained '.pt' files, e.g. (99 199 299 399)
eval_id1_list=(0)  # the ID index of the 1st person, e.g. (0 1 2 3 4)
eval_id2_list=(1)  # the ID index of the 2nd person, e.g. (0 1 2 3 4)
``` -->

**Testing**
```shell
bash ./02_start_test.sh "./weights/sd-v1-4-full-ema.ckpt" "./infer_images/example_prompt.txt" id_name "0 0 0 0" True 1 49 0.2 image_name.jpg 
```

The generated images are under `./outputs/id_name/`.

#### 3. Attribute Edit

Edit the prompt file `./infer_images/example_prompt.txt`, where `sks` denotes the first identity. `image_name.jpg` should be present inside 
`./aug_images/comparision/edited/` or else manually you have to change the root dir in code. There are some edits available in `all_delta_w_dict.json` file. You can check the keys and pass it as attr_name. Some attr present are - `smile, beard, bang, age70, gender, eyeglasses, yellow(asian), black, eyesclose, white`. 

**Testing attr edit**
```shell
bash ./02_start_test_pmm.sh "./weights/sd-v1-4-full-ema.ckpt" "./infer_images/example_prompt.txt" id_name "0 0 0 0" True 1 49 0.2 image_name.jpg attr_name
```
This will generate a gif and list of images with different edit strength.

<!-- #### 3. (Optional) Extracting ID Coefficients

Optionally, you can extract the coefficients for each identity by running:

```shell
bash ./03_extract.sh "./weights/sd-v1-4-full-ema.ckpt" "traininYYYY-MM-DDTHH-MM-SS_celebbasis"
```

The extracted coefficients or embeddings are under `./weights/ti_id_embeddings/`. -->

### TODO
- [x] release code
- [ ] multiple person generation

### BibTex

```tex
@article{yuan2023celebbasis,
  title={Inserting Anybody in Diffusion Models via Celeb Basis},
  author={Yuan, Ge and Cun, Xiaodong and Zhang, Yong and Li, Maomao and Qi, Chenyang and Wang, Xintao and Shan, Ying and Zheng, Huicheng},
  journal={arXiv preprint arXiv:2306.00926},
  year={2023}
}
```