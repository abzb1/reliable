## PGD Attack Results on CIFAR10 Dataset

#### Usage

Need python >= 3.10.   
Need more than 20476MiB VRAM.   
It takes about 10~20 minutes to train.

```bash
pip install -r requirements.txt
python test.py
```

### Test set Accuracy After Training

47.66%

### Attack Success Rate for Test set

slow..

Attack Success Rate (%, higher the better)
|    pgd_eps = 3e-2    | pgd_iters |  20   |  40   |  60   |  80   |  100  |  500  |
|----------------------|-----------|-------|-------|-------|-------|-------|-------|
| **pgd_alpha = 1e-2** |           | 43.39 | 43.45 | 43.44 | 43.44 | 43.45 | 43.47 |
| **pgd_alpha = 1e-3** |           | 35.10 | 42.99 | 43.30 | 43.47 | 43.51 | 43.63 |
| **pgd_alpha = 1e-4** |           |  4.45 |  8.92 | 13.14 | 17.21 | 20.71 | 43.23 |

#### decay alpha
```python
tau = iters // 2
alpha = alpha * math.exp(-i / tau)
```

Decaying alpha with respect to the step count actually led to worse results.

Attack Success Rate (%, higher the better)
|    pgd_eps = 3e-2    | pgd_iters |  50   |
|----------------------|-----------|-------|
| **pgd_alpha = 1e-2** |           | 43.23 |