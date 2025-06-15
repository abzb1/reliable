### α,β-CROWN

1. **α,β-CROWN** is a complete neural-network verification algorithm that uses **CROWN-based linear bound propagation** to compute initial bounds (α) and combines **β-stage optimization** with a divide-and-conquer **Branch-and-Bound (BaB)** procedure.  
2. The **α stage** quickly produces upper and lower bounds across the entire input domain with **GPU-accelerated CROWN bounds**, providing the starting point for the BaB search.  
3. The **β stage** jointly optimizes split-domain neuron constraints via **β parameters**, delivering tighter bounds than traditional LP-based methods.  
4. Repeating these two stages within **BaB** efficiently prunes away unnecessary regions, enabling complete **SAT/UNSAT** decisions even for CNNs with millions of parameters.  
5. The α,β-CROWN implementation ranked overall **first in VNN-COMP 2021 – 2024**, demonstrating its speed and verification accuracy.

### Environment setup

```bash
pip install uv
uv pip install -r requirements.txt
uv pip install -e "alpha-beta-CROWN/auto_LiRPA"
```

### Train model

`python train_model.py`

### Validate model

```python
python alpha-beta-CROWN/complete_verifier/abcrown.py \
--config resnet_18_cifar10.yaml \
> out.out 2>&1
```

### Result

Running the verifier **abcrown.py** on a **ResNet-18 (CIFAR-10)** model with an **L∞ attack radius ε = 2 / 255 (0.007843)** for **100 test samples**  

```
verified_status unsafe-pgd
verified_success True
Result: unsafe-pgd in 0.0222 seconds
############# Summary #############
Final verified acc: 0.0% (total 100 examples)
Problem instances count: 100 , total verified (safe/unsat): 0 , total falsified (unsafe/sat): 99 , timeout: 1
mean time for ALL instances (total 100):1.3208345236782664, max time: 120.99662661552429
mean time for verified UNSAFE instances (total 99): 0.11198827232977357, max time: 1.145664930343628
unsafe-pgd (total 99), index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
unknown (total 1), index: [81]
```
The preliminary **PGD attack** immediately found counter-examples for **99 samples**, so these were classified as **unsafe-pgd** (counter-example found by the attack) and the **Branch-and-Bound (BaB)** search was skipped.  
The remaining **one sample (index 81)** exceeded the configured timeout (120 s) and was left as **unknown**.  
The **final robust verified accuracy is 0 %**; no sample was proven safe (unsat).

It was a model that broke very easily.  
