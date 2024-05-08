### 개요
* 모델의 초기 파라미터가 동일한 것 확인.
* 정상적으로 학습이 되는 것 확인.
* AMP 사용
* Iteration 스케줄러 사용.
* torch 2.1 사용.

<br>

### A. 자동 혼합 정밀도(AMP)사용
* 자동 혼합 정밀도(AMP)를 사용하여 학습 - VRAM 차지하는 비율이 절반 가까이 감소.
> * Autocast 적용 - `with torch.cuda.amp.autocast()`
> * Gradient scaler로 Autocast로 인한 float16으로 조정을 보정 - `scaler = torch.cuda.amp.GradScaler()`
> * Gradient Clipping 사용 - AMP 사용 시, scaler 역산 후 수행되어야 함 - `scaler.unscale_(optimier)` clipping 전에 적용.

<br>

### B. Iteration 기반 스케줄러 사용
* 스캐쥴러는 다음 3가지 종류가 있음.
> 1. Epoch 기반 스케줄러
>> * StepLR, MultiStepLR, ExponentialLR 등
>> * Epoch의 끝에서 학습률 변경
> 2. Iteration 기반 스케줄러
>> * OneCycleLR, CosineAnnealingWarmRestarts 등
>> * 매 iteration 또는 mini-batch 처리 후 학습률 조정
> 3. 적응형 스케줄러
>> * 모델의 성능이나 메트릭을 기반으로 학습률 조정
>> * ReduceLROnPlateau(검증 데이터셋의 성능 향상이 멈출 때 학습률 감소)
> * 본 모델은 CosineAnnealingWarmRestarts을 사용하였으므로, iteration 안에서 스케줄러 조정.

<br>

### C. Torch 2.1 사용.
> * `torch.compile(model)`
> * `torch._dynamo.config.cache_size_limit = 512`

<br>

### D. Annotation 관련 주의 사항
> * bbox가 그림 외곽에 생성되는 경우가 있음.
> * bbox 생성 시, (x1, y1, x2, y2)가 (x_min, y_min, x_max, y_max)가 아닌 경우가 있음 - 반대로 Annotation한 경우.
