# Sundial

This is the official repository of [Sundial: A Family of Highly Capable  Time Series Foundation Models](https://arxiv.org/abs/2502.00]816) [[Intro]](https://mp.weixin.qq.com/s/y3sc2e2lmW1sqfnoK-ZdDA).

<p align="center">
<img src="./figures/cover.png" alt="" align=center />
</p>

## Updates

:triangular_flag_on_post: **News** (2025.06) Sundial has been accepted as **ICML 2025 Oral** (Top 1%). See you at Vancouver :)

:triangular_flag_on_post: **News** (2025.05) Get **1st MASE** on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) Benchmark.

:triangular_flag_on_post: **News** (2025.05) Released a **trillion-scale** pre-trained model on [HuggingFace](https://huggingface.co/thuml/sundial-base-128m). A quickstart is provided [here](./examples/quickstart_zero_shot_generation.ipynb).

:triangular_flag_on_post: **News** (2025.02) Get **1st MSE/MAE** zero-shot performance on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) datasets.

## Introduction

Sundial is a family of **generative** time series foundation models, which is pre-trained on TimeBench (**10^12** time points). The model can be applied for both **point** and **probabilistic** forecasting.

Not only the mean or quantiles, you can estimate anything about the predictive distribution with raw generated samples.

We propose **TimeFlow Loss** to predict next-patch’s distribution, allowing Transformers to be trained **without discrete tokenization** and make **non-deterministic predictions**.

<p align="center">
<img src="./figures/compare.png" alt="" align=center />
</p>

## Quickstart

We release a [HuggingFace model](https://huggingface.co/thuml/sundial-base-128m), which can make zero-shot predictions on CPU within seconds! 🚀

> Inference Time on Apple M1 Pro CPU (16 GB)

| Lookback | Forcast | # Generated | Wall-Clock Time | Accelerate By         |
| --------------- | ----------------- | ------------------- | -------------- | --------------         |
| 672             | 16                | 1                   | 249ms          |       -                |
| 2880            | 16                | 1                   | 510ms          | FlashAttention         |
| 2880            | 720               | 1                   | 510ms          | Multi-Patch Prediction |
| 2880            | 1440              | 1                   | 789ms          | KV Cache               |
| 2880            | 720               | 20                  | 949ms          | Shared Condition       |

All you need is a network and a HuggingFace account!

```
pip install transformers==4.40.1
```

```
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
# supports different lookback/forecast lengths
model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True) 

# prepare input
batch_size, lookback_length = 1, 2880 
seqs = torch.randn(batch_size, lookback_length)

# Note that Sundial can generate multiple probable predictions
forecast_length = 96 
num_samples = 20

output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)

# use raw predictions for mean/quantiles/confidence-interval estimation
print(output.shape) 
```

More examples of predicting quantiles or confidence intervals are provided in this [notebook](https://github.com/thuml/Sundial/blob/main/examples/quickstart_zero_shot_generation.ipynb). Please raise your valuable suggestions [here](https://huggingface.co/thuml/sundial-base-128m/discussions/new), we 'd like to solve it ASAP 🤗.



## Architecture

<p align="center">
<img src="./figures/arch.png" alt="" align=center />
</p>

> Intuitively, Sundial can be viewed as an **ARMA** model (Auto-Regression and Moving-Average). Transformer learns auto-regressive token representations. Conditioned on them, TimeFlow transforms random noises into non-deterministic predictions.

## Model Configurations

We have currently built three different sizes of Sundial. Model configurations are provided here:

<p align="center">
<img src="./figures/config.png" alt="" align=center />
</p>

## Evaluation

We evaluate Sundial (Base) with advanced time series foundation models on well-recognized benchmarks:

- [GIFT-Eval (1st MASE)](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/3BxatwayhK5GAoqMf1oHv.png) [[Leaderboard]](https://huggingface.co/spaces/Salesforce/GIFT-Eval).
- [Time-Series-Library (1st MSE/MAE)](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/5VqnFwWTWoYz877Zkluiw.png).
- [FEV Leaderboard](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/mrKL9QmX-aX8rCiwxKgmA.png).

## Exciting News ✨

 Code for fine-tuning is on its way and will be available soon! Stay tuned for updates!

## Citation

If you find this repo helpful, please cite our paper. 


```
@article{liu2025sundial,
  title={Sundial: A Family of Highly Capable Time Series Foundation Models},
  author={Liu, Yong and Qin, Guo and Shi, Zhiyuan and Chen, Zhi and Yang, Caiyin and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2502.00816},
  year={2025}
}
```

## Acknowledgment

We appreciate the following resources a lot for their valuable code and datasets:

- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Large-Time-Series-Model & UTSD (https://github.com/thuml/Large-Time-Series-Model)
- Timer-XL (https://github.com/thuml/Timer-XL)
- LoTSA Data (https://huggingface.co/datasets/Salesforce/lotsa_data)
- Chronos Datasets (https://huggingface.co/datasets/autogluon/chronos_datasets)

## Contact

If you have any questions or want to use the code, feel free to contact:

* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qinguo24@mails.tsinghua.edu.cn)

## License

This model is licensed under the Apache-2.0 License.
