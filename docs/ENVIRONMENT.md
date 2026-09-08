# Environment

Use an isolated environment and retain the package inventory with every run.
The package requirements specify installation ranges. For comparisons with
recorded runs, match the package versions, CUDA runtime and GPU listed below.

## Recorded experiment environments

| Component | Principal seed-42 runs | August sensitivity runs |
|---|---|---|
| Python | 3.12.3 | 3.12.3 |
| PyTorch | 2.8.0+cu128 | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 | 0.23.0+cu128 |
| CUDA runtime | 12.8 | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 | NVIDIA GeForce RTX 5090 |
| timm | 1.0.27 | 1.0.28 |
| NumPy | 2.3.2 | 2.3.2 |
| pandas | 3.0.3 | 3.0.5 |
| scikit-learn | 1.9.0 | 1.9.0 |

The table records versions saved with the experiments. Obtain matching packages
from their original indexes and check the CUDA wheel's compatibility with the
target environment.

## Optional B6-M backend

B6-M additionally recorded `mamba-ssm==2.3.2.post1`. The associated installation
used `causal-conv1d==1.6.2.post1` and `transformers==4.57.1`. Use a separate
Linux/CUDA environment and follow the official
[Mamba installation instructions](https://github.com/state-spaces/mamba/tree/v2.3.2.post1).
Match Python, PyTorch, CUDA, C++ ABI and GPU architecture when selecting wheels.

```bash
python -m pip install -r requirements-mamba.txt --no-build-isolation
python -m pip check
python scripts/21_mamba_backend_smoke.py
```

Install the complete dependency set in the isolated environment, resolve any
reported PyTorch/Triton version conflicts, and run both checks above. The smoke
test exercises forward and backward CUDA execution. Backend selection is explicit:
CSSA-enabled principal configurations use axial convolution, and B6-M uses `mamba_ssm`.
