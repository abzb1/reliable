## UnTargeted FGSM Attack Results on MNIST Dataset

#### Usage

Need python >= 3.10.   
Need more than 23000MiB VRAM.   
It takes about 10~20 minutes to train.

```bash
pip install -r requirements.txt
python test.py
```

### Test set Accuracy After Training

97.89%

### Attack Success Rate for Test set

It definitely feels easier to succeed compared to the targeted attack.
However, it is fairly well defended against small epsilon values.

![image](https://github.com/user-attachments/assets/3c427559-a200-4b9c-84ea-def371466768)

| fgsm_eps |  MLP   |
|----------|--------|
| 3e-2     | 30.22% |
| 1e-2     |  3.45% |
| 7e-3     |  1.95% |
| 5e-3     |  1.35% |
| 3e-3     |  0.73% |
| 1e-3     |  0.24% |
| 1e-4     |  0.03% |
| 1e-5     |  0.00% |
