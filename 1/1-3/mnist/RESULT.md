## PGD Attack Results on MNIST Dataset

#### Usage

Need python >= 3.10.   
Need more than 620MiB VRAM.   
It takes about 10~20 minutes to train.

```bash
pip install -r requirements.txt
python test.py
```

### Test set Accuracy After Training

97.89%

### Attack Success Rate for Test set

When alpha is large, the attack success rate increases quickly at first, but it seems to fall short of reaching the optimum as the number of iterations grows.
It feels similar to the effect of a learning rate.
This makes me wonder if introducing a scheduler could help.

Attack Success Rate (%, higher the better)
| pgd_eps = 3e-2 | pgd_iters | 5     | 10    | 15    | 20    | 25    | 30    | 35    | 40    | 500   |
|----------------|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| **pgd_alpha = 1e-2** |             | 33.45 | 34.41 | 34.6  | 34.65 | 34.67 | 34.67 | 34.69 | 34.69 | 34.69 |
| **pgd_alpha = 1e-3** |             | 1.35  | 3.51  | 7.02  | 12.28 | 20.8  | 32.81 | 33.24 | 33.53 | 34.71 |
| **pgd_alpha = 1e-4** |             | 0.11  | 0.25  | 0.4   | 0.53  | 0.62  | 0.73  | 0.87  | 1.03  | 34.05 |