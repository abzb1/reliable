## Targeted FGSM Attack Results on CIFAR10 Dataset

#### Usage

Need python >= 3.10.   
Need more than 20484MiB VRAM.   
It takes about 10 minutes to train.

```bash
pip install -r requirements.txt
python test.py
```

### Test set Accuracy After Training

| Model | Accuracy |
|-------|----------|
| CNN   | 47.66%   |

---

### Attack Success Rate for Test set

| fgsm_eps |  CNN  |
|----------|-------|
| 2e-2     | 22.51% |
| 1e-2     | 13.23% |
| 9e-3     | 12.49% |
| 8e-3     | 11.70% |
| 7e-3     | 10.83% |
| 6e-3     | 9.94% |
| 5e-3     | 9.22% |
| 4e-3     | 8.35% |
| 3e-3     | 7.50% |
| 2e-3     | 6.85% |
| 1e-3     | 6.20% |