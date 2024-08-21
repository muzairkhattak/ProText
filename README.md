# ProText: Prompt Learning with Text Only Supervision


> [**Learning to Prompt with Text Only Supervision for Vision-Language Models**](https://arxiv.org/abs/2401.02418) <br>
> [Muhammad Uzair Khattak](https://muzairkhattak.github.io/), [Muhammad Ferjad Naeem](https://ferjad.github.io/), [Muzammal Naseer](https://scholar.google.com/citations?user=tM9xKA8AAAAJ&hl=en&oi=ao), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en), [Federico Tombari](https://scholar.google.de/citations?user=TFsE4BIAAAAJ&hl=en&oi=ao)



[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.02418)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://muzairkhattak.github.io/ProText/)
[![video](https://img.shields.io/badge/video-teaser-orange)](https://youtu.be/HecFYi-WpFI)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1H7u4QJwLYijbLAL0eAVJco64f59OokQQ/view?usp=sharing)


Official PyTorch implementation

<hr />


# :rocket: News
* **(April 21, 2024)**
  * Paper accepted at CVPR Workshop on What is Next in Multimodal Foundation Models? 2024. 
* **(Jan 05, 2023)**
  * Pre-trained models and evaluation codes for ProText are released.
  * Training codes for [ProText](configs/trainers/ProText) are released.
  * This repository also supports [CuPL (ICCV'23)](scripts/zsclip_cupl), [PromptSRC (ICCV'23)](configs/trainers/PromptSRC), [MaPle (CVPR'23)](configs/trainers/MaPLe),
[CoOp (IJCV'22)](configs/trainers/CoOp), [Co-CoOp (CVPR'22)](configs/trainers/CoCoOp) 
architectures.
<hr />

## Highlights

![main figure](docs/main_figure.png)
> <p align="justify"> <b> <span style="color: blue;">Left</span></b>:
> Existing methods improve CLIP's generalization by learning prompts with image
> supervision or using non-transferable prompt ensembling with LLM knowledge. 
> In contrast, our approach, ProText, effectively learns prompts with LLM knowledge based text-only supervision
> which are transferable to new datasets and classes.. <b><span style="color: blue;">Right</span></b>:
> Without using any images for supervision, ProText with text-only training improves over CLIP, 
> CuPL, and prior 16-shot image-supervised methods in challenging cross-dataset transfer settings.
> Prompt ensembling based CuPL performs same as CLIP as it cannot transfer class specific LLM 
> templates to cross-datasets.




> **<p align="justify"> Abstract:** *Foundational vision-language models such as CLIP are becoming a
> new paradigm in vision, due to their excellent generalization abilities. However, adapting these 
> models for downstream tasks while maintaining their generalization remains a challenge.
> In literature, one branch of methods adapts CLIP by learning prompts using visual
> information. While effective, most of these works require labeled data which is not practical,
> and often struggle to generalize towards new datasets due to over-fitting on the source data.
> An alternative approach resorts to training-free methods by generating class descriptions from 
> large language models (LLMs) and perform prompt ensembling. However, these methods often 
> generate class specific prompts that cannot be transferred to other classes, which incur 
> higher costs by generating LLM descriptions for each class separately. In this work,
> we propose to combine the strengths of these both streams of methods by learning 
> prompts using only text data derived from LLMs. As supervised training of prompts
> is not trivial due to absence of images, we develop a training approach that 
> allows prompts to extract rich contextual knowledge from LLM data. Moreover, 
> with LLM contextual data mapped within the learned prompts, it enables zero-shot 
> transfer of prompts to new classes and datasets potentially cutting the LLM prompt 
> engineering cost. To the best of our knowledge, this is the first work that learns generalized 
> prompts using text only data. We perform extensive evaluations on 4 benchmarks where our
> method improves over prior ensembling works while being competitive to those utilizing labeled 
> images. Our code and pre-trained models are publicly available.* </p>

## A Text Only Prompt Learning framework Vision-Language Models

We propose ProText: **Pro**mpt Learning with **Text** Only Supervision, which leverages text data to learn transferable prompts to enhance CLIP's generalization for visual recognition tasks.

**Main contributions :**
1) **Text Only Prompt Learning Approach:** We develop a new approach for prompt learning in Vision-Language models without relying on visual samples for visual recognition tasks.
2) **Learning Prompts with Contextual Mapping:**  We introduce a training strategy for prompts to learn a mapping function that embeds rich and generalized contextual knowledge from Large Language Models (LLMs) based text data within the prompts.
3) **LLM Prompts Transferability:** With LLM contextual data mapped within the learned prompts, it enables zero-shot transfer of prompts to new classes and datasets, potentially reducing the LLM prompt engineering and serving costs.


## :ballot_box_with_check: Supported Methods

| Method                    | Paper                                                                        |               Configs               |        Training Scripts         |
|---------------------------|:-----------------------------------------------------------------------------|:-----------------------------------:|:-------------------------------:|
| ProText                   | [arXiv](https://arxiv.org/abs/2401.02418)                                                                    |  [link](configs/trainers/ProText/)  |    [link](scripts/promptsrc)    |
| CuPL (Baseline)           | [arXiv](https://arxiv.org/abs/2209.03320)                                    |  [link](configs/trainers/ProText/)  |   [link](scripts/zsclip_cupl)   |
| PromptSRC                 | [arXiv](https://arxiv.org/abs/2307.06948)                                    | [link](configs/trainers/PromptSRC/) |    [link](scripts/promptsrc)    |
| Independent V-L Prompting | -                                                                            |   [link](configs/trainers/IVLP/)    | [link](scripts/independent-vlp) |
  | MaPLe                     | [CVPR 2023](https://arxiv.org/abs/2210.03117)                                |    [link](configs/trainers/CoOp)    |      [link](scripts/maple)      |
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134)                                |    [link](configs/trainers/CoOp)    |      [link](scripts/coop)       |
| Co-CoOp                   | [CVPR 2022](https://arxiv.org/abs/2203.05557)                                |   [link](configs/trainers/CoCoOp)   |     [link](scripts/cocoop)      |

<hr />

## Results

### ProText fares well in comparison with Prompt Ensembling methods
First we show that with same amount of text data, learning contextual prompts with text-only supervision improves CLIP performance against prompt ensembling techniques. 

| Method                                                    | ImageNet Acc. |    
|-----------------------------------------------------------|:--------------|
| [CLIP](https://arxiv.org/abs/2103.00020) (ICML'21)                                        | 66.72         | 
| [DCLIP](https://arxiv.org/abs/2210.07183) (ICLR'23)       | 68.03         | 
| [Waffle CLIP](https://arxiv.org/abs/2306.07282) (ICCV'23) | 68.34         | 
| [CuPL](https://arxiv.org/abs/2209.03320) (ICCV'23)        | 69.62         |   
  | **ProText (Ours)**                                            | **70.22**     |   


### ProText addresses the transferability limitations of LLM based Prompt Ensembling methods

With the contextual LLM information mapped with in the prompts, ProText enables the transferability
of learned prompts to new classes and improves over CuPL.

| Method                                   | Base Acc. | Novel Acc. |  HM   |
|------------------------------------------|:---------:|:----------:|:-----:|
| [CLIP](https://arxiv.org/abs/2103.00020) |   69.34   |   74.22    | 71.70 |  
| [CuPL](https://arxiv.org/abs/2209.03320) |   72.56   |   74.22    | 73.38 | 
| **ProText (ours)**                           |   **72.95**   |   **76.98**    | **74.91** |  

Results reported above show accuracy for base and novel classes for across 11 recognition datasets.


### ProText performs favourably well in Cross-dataset benchmark 
ProText with text-only training improves over CLIP, CuPL, and prior 16-shot image-supervised methods in challenging cross-dataset transfer settings. Prompt ensembling based CuPL performs same as CLIP as it cannot transfer class specific LLM templates to cross-datasets.

| Method                                                                              | Supervision Type | Avg Acc. |
|-------------------------------------------------------------------------------------|:----------------:|:--------:|
| [CoOP](https://arxiv.org/abs/2109.01134)                                            |  labeled images  |  63.88   |  
| [CoCoOp](https://arxiv.org/abs/2203.05557)                                          |  labeled images  |  65.74   |   
| [MaPLe](https://arxiv.org/abs/2210.03117)                                           |  labeled images  |  66.30   | 
| [PromptSRC](https://arxiv.org/abs/2307.06948)                                       |  labeled images  |  65.81   | 
| [CLIP](https://arxiv.org/abs/2103.00020) / [CuPL](https://arxiv.org/abs/2209.03320) |   Text Prompts   |  65.15   | 
| **ProText (ours)**                                                                      |       Text Prompts       |  **67.23**   | 

Models are trained on ImageNet-1k data and evaluated on 10 cross-datasets. 


## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data Preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

## Model Zoo

### Vision-Language prompting methods
| Method  (configs)                    |                                                               Model checkpoints                                                                |
|--------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------:|
| [CuPL](configs/trainers/ProText/)    |                                                     [Direct evaluation](docs/CuPL_CLIP.md)                                                     |
| [ProText](configs/trainers/ProText/) | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muzammal_naseer_mbzuai_ac_ae/ErAYiYg5N0xKgJSDm-zeUKQB_6OAoIYb2giZvoLcAFoZSg?e=lxSEhb) |


## Quick Start 
ProText can be easily trained on text-data with simple scripts.
For example, to train ProText on ImageNet-1k text data, you can run the following command:
```
sh scripts/protext/fully_supervised.sh imagenet output/experiment/path
```
Please refer to the [TRAIN.md](docs/TRAIN.md) for detailed instructions on training ProText and producing results for CuPL baseline.


For evaluation using pretrained weights, you can run the following command:
```
sh scripts/protext/fully_supervised_and_dg.sh imagenet path/to/pretrained/weights
```

Please refer to the [EVAL.md](docs/EVAL.md) for detailed instructions on using the evaluation scripts and reproducing the official results using our pre-trained models.


<hr />

## Citation
If you find our work, this repository, or pretrained models useful, please consider citing:
```bibtex
@article{Khattak2024ProText,
    title={Learning to Prompt with Text Only Supervision for Vision-Language Models},
    author={khattak, Muhammad Uzair and Ferjad, Muhammad and Muzzamal, Naseer and Gool, Luc Van and Tombari, Federico},
    journal={arXiv:2401.02418},
    year={2024}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at uzair.khattak@mbzuai.ac.ae.


## Acknowledgements

Our code is based on [PromptSRC](https://github.com/muzairkhattak/multimodal-prompt-learning). We have additionally borrowed partial code from [CuPL](https://github.com/sarahpratt/CuPL) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

