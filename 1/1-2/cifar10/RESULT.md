## UnTargeted FGSM Attack Results on CIFAR10 Dataset

#### Usage

Need python >= 3.10.   
Need more than 620MiB VRAM.   
It takes about 10 minutes to train.

```bash
pip install -r requirements.txt
python test.py
```

### Test set Accuracy After Training

47.66%

### Attack Success Rate for Test set

It definitely feels easier to succeed compared to the targeted attack.

| fgsm_eps |  CNN   |
|----------|--------|
| 3e-2     | 39.72% |
| 1e-2     | 20.04% |
| 7e-3     | 14.75% |
| 5e-3     | 10.82% |
| 3e-3     |  6.57% |
| 1e-3     |  2.12% |
| 1e-4     |  0.16% |