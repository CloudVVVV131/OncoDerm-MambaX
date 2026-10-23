# Third-Party Software and Data

Original source code is licensed under the [MIT License](LICENSE). Dependencies
are installed separately and retain their upstream licenses and copyright
notices, as linked below.

| Component | Upstream source / license | Use |
|---|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) | Tensors and training |
| torchvision | [pytorch/vision](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE) | Transforms and models |
| timm | [Apache-2.0, Ross Wightman and contributors](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) | ConvNeXt and comparators |
| mamba-ssm | [Apache-2.0, Tri Dao and Albert Gu](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/LICENSE) | Optional B6-M backend |
| causal-conv1d | [BSD-3-Clause](https://github.com/Dao-AILab/causal-conv1d/blob/main/LICENSE) | Optional CUDA extension |
| THOP | [Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter/blob/master/LICENSE) | Operation counting |
| NumPy, pandas, scikit-learn | [NumPy](https://numpy.org/doc/stable/license.html), [pandas](https://pandas.pydata.org/docs/getting_started/overview.html#license), [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | Tables and metrics |
| matplotlib, seaborn | [matplotlib](https://matplotlib.org/stable/project/license.html), [seaborn](https://github.com/mwaskom/seaborn/blob/master/LICENSE.md) | Figures |
| Pillow, PyYAML, tqdm, einops, remotezip | Licenses in their package distributions | Image, configuration and data utilities |

The DermaMamba reference repository is
[Glow945/Skin_Lesions](https://github.com/Glow945/Skin_Lesions/tree/2f5542bf30b9c5f6e0cecc6b5ac498cce76ca567).
The comparator implementation is outside this source distribution.

Dataset and pretrained-weight terms are separate from software licenses. Obtain
HAM10000 and ISIC data through [DATA.md](docs/DATA.md). The
[ISIC download page](https://challenge.isic-archive.com/data/) supplies
dataset-specific terms; SIIM-ISIC 2020 uses CC BY-NC 4.0, including its
noncommercial condition. Pretrained checkpoints obtained through timm or
torchvision retain their source-specific terms for use and redistribution.
