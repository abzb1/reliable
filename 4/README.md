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

검증기 abcrown.py가 ResNet-18 (CIFAR-10) 모델을 L∞-공격 반경 ε = 2 / 255(0.007843)로 100 개의 테스트 샘플에 대해 돌린 결과

```
Checking and Saving Counterexample in check_and_save_cex

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

사전 단계로 실행된 PGD 공격이 99 개의 반례를 즉시 찾아냈으므로 이들 샘플은 unsafe-pgd(공격으로 반례가 발견됨)로 분류되었고, 분기-한계(BaB) 탐색은 수행되지 않았음
나머지 1 개(인덱스 81)는 설정-상 타임아웃(120 초)을 넘겨 unknown 으로 남음
최종 검증 정확도(robust verified accuracy)는 0 %, 안전(unsat)으로 증명된 샘플은 없음

굉장히 쉽게 깨진 모델이었다.

### α,β-CROWN

1.	α,β-CROWN은 CROWN 기반 선형 경계 전파를 활용해 초기 한계(α)​를 계산하고, 분할-정복 Branch-and-Bound에 β 단계 최적화를 결합한 완전 신경망 검증 알고리즘이다.  ￼ ￼
2.	α 단계는 GPU 가속 CROWN 경계로 전체 입력 공간의 상·하한을 빠르게 산출해 BaB 탐색의 출발점을 마련한다.  ￼
3.	β 단계는 각 분할 도메인에서 뉴런 분할 제약을 ‘β 파라미터’로 공동 최적화하여 LP-기반 기법보다 타이트한 경계를 제공한다.  ￼ ￼
4.	이 두 단계를 반복하는 BaB 과정은 불필요한 영역을 효율적으로 가지치기하며, 수백만 파라미터 CNN도 SAT/UNSAT을 완전 판정할 수 있게 했다.  ￼ ￼
5.	구현체 α,β-CROWN은 국제 대회 VNN-COMP 2021–2024에서 연속 종합 1위를 차지해 속도와 검증 정확성을 입증했다.