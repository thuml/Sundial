# Sundial

Sundial: A Familiy of Highly Capable  Time Series Foundation Models [[Paper]](https://arxiv.org/abs/2502.00816).

:triangular_flag_on_post: **News** (2025.05) Released a **trillion-scale** pre-trained model on [HuggingFace](https://huggingface.co/thuml/sundial-base-128m). A quickstart usage is provided [here](./quickstart_zero_shot.ipynb).

:triangular_flag_on_post: **News** (2025.05) **Ranked 1st MASE** on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) Benchmark.

:triangular_flag_on_post: **News** (2025.05) Sundial has been accepted as **ICML 2025 Spotlight**. See you at Vancouver :)

## Introduction

Sundial is a familiy of **generative** time series foundation models. The model can make zero-shot predictions for both **point** and **probabilistic** forecasting.


💡 We propose TimeFlow Loss to predict next-patch’s distribution, allowing Transformers to be trained **without discrete tokenization** and make **multiple probable predictions**.

💪 We release Sundial, a family of **scalable** and **efficient** time series foundation models pre-trained on **1 trillion** time points, utilizing our enhanced Transformer.

🏆 Sundial achieves **state-of-the-art** zero-shot performance on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), [FEV](https://huggingface.co/spaces/autogluon/fev-leaderboard), and [TSLib](https://github.com/thuml/Time-Series-Library).

<p align="center">
<img src="./figures/motivation.png" alt="" align=center />
</p>

## Quickstart

We release checkpoint and deft model wrapper to make zero-shot predictions on your customized data:

```
pip install transformers==4.40.1
```

```
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)

# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length)

# generate forecast
prediction_length = 96
num_samples = 20
output = model.generate(seqs, max_new_tokens=prediction_length, num_samples=num_samples)

print(output.shape) # generate 20 probable predictions
```

More examples for predicting quantiles or confidence intervals is provided [here](https://github.com/thuml/Sundial/blob/main/examples/quickstart_zero_shot.ipynb).

## Architecture

<p align="center">
<img src="./figures/arch.png" alt="" align=center />
</p>

Input time series is divided into patch tokens, which are embedded from original continuous values. The patch embeddings are fed into a decoder-only Transformer, a speedup version that learns token representations via causal self-attention. The model is optimized using **TimeFlow** Loss.


## TimeFlow Loss

We propose TimeFlow Loss, a parameterized loss function that models per-token probability distribution conditioned on token representations, and generates multiple plausible predictions under the **flow-matching** framework.

This optimization objective operates on original values and facilitates patch-level generation for quick inference, which is highly compatible with continuous-valued modalities, such as time series.


> Training

$$
\mathcal{L}_{\text {TimeFlow }}=\sum_i^N \| \text { FM-Net }\left(\mathbf{y}_i^{(t)}, t, \mathbf{h}_i\right)-\left(\mathbf{y}_i-\mathbf{y}_i^{(0)}\right) \|^2
$$

> Sampling

<p align="center">
<img src="./figures/tf_infer.png" alt="" align=center />
</p>


## Evaluation

We evaluate Sundial with advanced time series foundation models in these aspects:

### Peformance

- [Time-Series-Library](./figures/tslib_res.png)
- [FEV Leaderboard](./figures/fev_res.png)
- [GIFT-Eval](./figures/gift_res.png)

### Inference Speed

- [FEV Leaderboard](./figures/fev_eff.png)


### Scalability

- [Pre-training](./figures/train_scale.png)
- [Test-Time](./figures/test_scale.png)
  
## Showcases

- [Time-Series-Library](./figures/tslib_case.png)
- [FEV Leaderboard]((./figures/fev_case.png))
- [GIFT-Eval (Code)](./notebook)

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

For our previous work, please refer to [Timer](https://github.com/thuml/Large-Time-Series-Model) and [Timer-XL](https://github.com/thuml/Timer-XL).


## Acknowledgment

We appreciate the following GitHub repos a lot for their valuable code and efforts:

- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Large-Time-Series-Model (https://github.com/thuml/Large-Time-Series-Model)
- Timer-XL (https://github.com/thuml/Timer-XL)

## Contact

If you have any questions or want to use the code, feel free to contact:

* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qinguo24@mails.tsinghua.edu.cn)
