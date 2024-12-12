[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://github.com/Vchitect/Evaluation-Agent)
[![Project Page](https://img.shields.io/badge/Evaluation-Website-green?logo=googlechrome&logoColor=green)](https://github.com/Vchitect/Evaluation-Agent-project)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FEvaluation-Agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


This repository contains the implementation of the following work:
> **Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models**<br>
> [Fan Zhang](https://github.com/zhangfan-p)<sup>∗</sup>, [Shulin Tian](https://shulin16.github.io/)<sup>∗</sup>, [Ziqi Huang](https://ziqihuangg.github.io/)<sup>∗</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/index.html)<sup>+</sup>, [Ziwei Liu](https://liuziwei7.github.io/)<sup>+</sup><br>




<a name="overview"></a>
## :mega: Overview
![Framework](./assets/fig_framework.jpg)
We propose the **Evaluation Agent** framework, which simulates the human evaluation process. It can handle free-form evaluation requests from users and efficiently assess visual generative models through dynamic, multi-step evaluations, ultimately providing detailed analysis. It offers four key advantages: <b>1) dynamic assessment</b> enables efficient and detailed evaluations, <b>2) promptable evaluation</b> tailored to diverse user needs, <b>3) detailed and explainable</b> evaluation results rather than a single numerical score, and <b>4) scalability</b> to accommodate various visual generative models and evaluation tools.




<a name="installation"></a>
## :hammer: Installation

1. Clone the repository.

```bash
git clone https://github.com/Vchitect/Evaluation-Agent.git
cd Evaluation-Agent
```

2. Install the environment.
```bash
conda create -n eval_agent python=3.10
conda activate eval_agent
pip install -r requirements.txt
```



<a name="usage"></a>
## Usage

First, you need to configure the `open_api_key`. You can do it as follows:
```
export OPENAI_API_KEY="your_api_key_here"
```

### Evaluation of Open-ended Questions on T2I Models


```
python open_ended_eval.py --user_query $USER_QUERY --model $MODEL
```
- `$USER_QUERY` can be any question regarding the model’s capabilities, such as ‘How well does the model generate trees in anime style?’
- `$MODEL` refers to the image generation model you want to evaluate. Currently, we support four models: [SD-14](https://huggingface.co/CompVis/stable-diffusion-v1-4), [SD-21](https://huggingface.co/stabilityai/stable-diffusion-2-1), [SDXL-1](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and [SD-3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers). You can integrate new models in the following path: `./eval_agent/eval_models/`


### Evaluation Based on the VBench Tools on T2V Models

#### Preparation

1. Configure the VBench Environment

- You need to configure the VBench environment on top of the existing environment. For details, refer to [VBench](https://github.com/Vchitect/VBench).

2. Prepare the Model to be Evaluated

- Download the weights of the target model for evaluation and place them in `./eval_agent/eval_models/{model_name}/checkpoints/`. 

- Currently, we support four models: [latte](https://github.com/Vchitect/Latte/tree/main), [modelscope](https://modelscope.cn/models/iic/text-to-video-synthesis/summary), [videocrafter-0.9](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254), and [videocrafter-2](https://github.com/AILab-CVC/VideoCrafter). These models may also have specific environment requirements. For details, please refer to the respective model links.

#### Command

```
python eval_agent_for_vbench.py --user_query $USER_QUERY --model $MODEL
```
- `$USER_QUERY` need to be related to the 15 dimensions of VBench. These dimensions are: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`, `object_class`, `multiple_objects`, `human_action`, `color`, `spatial_relationship`, `scene`, `temporal_style`, `appearance_style`, and `overall_consistency`.
- `$MODEL` refers to the video generation model you want to evaluate.



### Evaluation Based on the T2I-CompBench Tools on T2I Models

#### Preparation

1. Configure the T2I-CompBench Environment

- You need to configure the T2I-CompBench environment on top of the existing environment. For details, refer to [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench/tree/6ea770ada4eea55fa7b09caa2d2fb63fe4d6bf8f).

2. Prepare the Model to be Evaluated

#### Command

```
python eval_agent_for_t2i_compbench.py --user_query $USER_QUERY --model $MODEL
```
- `$USER_QUERY` need to be related to the 4 dimensions of T2I-CompBench. These dimensions are: `color_binding`, `shape_binding`, `texture_binding`, `non-spatial relationship`.
- `$MODEL` refers to the image generation model you want to evaluate.





## Open-Ended User Query Dataset
We propose the **Open-Ended User Query Dataset**, developed through a user study. As part of this process, we gathered questions from various sources, focusing on aspects users consider most important when evaluating new models. After cleaning, filtering, and expanding the initial set, we compiled a refined dataset of 100 open-ended user queries.

Check out the details of the [open-ended user query dataset](https://github.com/Vchitect/Evaluation-Agent/tree/main/dataset) 

![statistic](./assets/open_dataset_stats.png)
The three graphs give an overview of the distributions and types of our curated open queries set. Left: the distribution of question types, which are categorized as `General` or `Specific`. Middle: the distribution of the ability types, which are categorized as `Prompt Following`, `Visual Quality`, `Creativity`, `Knowledge` and `Others`. Right: the distribution of the content categories, which are categorized as `History and Culture`, `Film and Entertainment`, `Science and Education`, `Fashion`, `Medical`, `Game Design`, `Architecture and Interior Design`, `Law`.


## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{zhang2024evaluationagent,
    author = {Zhang, Fan and Tian, Shulin and Huang, Ziqi and Qiao, Yu and Liu, Ziwei},
    title = {Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models},
    year = {2024}
}
```
