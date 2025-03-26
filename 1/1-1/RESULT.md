## FGSM Attack Results on MNIST Dataset

#### Usage

Need python >= 3.10.   
Need more than 23000MiB VRAM.   
It takes about 10~20 minutes to train.

```bash
pip install -r requirements.txt
python test.py
```

### Test set Accuracy After Training (model params ≈ 300K)

| Model | Accuracy |
|-------|----------|
| ViT   | 97.20%   |
| CNN   | 97.65%   |
| MLP   | 97.66%   |

---

### Attack Success Rate for Test set

Observation of FGSM success rates on models with similar parameter sizes and comparable accuracies reveals structural characteristics.
Although it was expected that ViT would be more robust to FGSM attacks due to its ability to capture global features, the results show that at larger epsilon values (≥ 7e-3), ViT is actually more vulnerable than CNN.
However, at smaller epsilon values, ViT exhibits lower attack success rates compared to CNN, aligning with the intuition that it is more robust to small perturbations.
For the MLP, a consistently similar or higher level of attack success rate is observed across all epsilon values.

![FGSM Attack Success Rate vs  Epsilon](https://github.com/user-attachments/assets/58eae6b7-a9ba-46d0-aba6-387d3db004da)

| fgsm_eps | ViT   | CNN   | MLP   |
|----------|-------|-------|-------|
| 2e-2     | 0.94% | 0.55% | 2.07% |
| 1e-2     | 0.50% | 0.38% | 0.66% |
| 9e-3     | 0.46% | 0.37% | 0.59% |
| 8e-3     | 0.43% | 0.35% | 0.53% |
| 7e-3     | 0.40% | 0.31% | 0.32% |
| 6e-3     | 0.34% | 0.35% | 0.40% |
| 5e-3     | 0.33% | 0.33% | 0.35% |
| 4e-3     | 0.31% | 0.33% | 0.32% |
| 3e-3     | 0.28% | 0.35% | 0.35% |
| 2e-3     | 0.22% | 0.30% | 0.30% |
| 1e-3     | 0.21% | 0.29% | 0.26% |
| 9e-4     | 0.21% | 0.29% | 0.26% |
| 8e-4     | 0.21% | 0.29% | 0.25% |
| 7e-4     | 0.21% | 0.29% | 0.25% |
| 6e-4     | 0.21% | 0.29% | 0.25% |
| 5e-4     | 0.21% | 0.28% | 0.25% |
| 5e-5     | 0.21% | 0.28% | 0.21% |
| 5e-6     | 0.21% | 0.28% | 0.20% |
