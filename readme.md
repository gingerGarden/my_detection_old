{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 개요\n",
    "* 모델의 초기 파라미터가 동일한 것 확인.\n",
    "* 정상적으로 학습이 되는 것 확인.\n",
    "* AMP 사용\n",
    "* Iteration 스케줄러 사용.\n",
    "* torch 2.1 사용.\n",
    "\n",
    "<br>\n",
    "\n",
    "### A. 자동 혼합 정밀도(AMP)사용\n",
    "* 자동 혼합 정밀도(AMP)를 사용하여 학습 - VRAM 차지하는 비율이 절반 가까이 감소.\n",
    "> * Autocast 적용 - `with torch.cuda.amp.autocast()`\n",
    "> * Gradient scaler로 Autocast로 인한 float16으로 조정을 보정 - `scaler = torch.cuda.amp.GradScaler()`\n",
    "> * Gradient Clipping 사용 - AMP 사용 시, scaler 역산 후 수행되어야 함 - `scaler.unscale_(optimier)` clipping 전에 적용.\n",
    "\n",
    "<br>\n",
    "\n",
    "### B. Iteration 기반 스케줄러 사용\n",
    "* 스캐쥴러는 다음 3가지 종류가 있음.\n",
    "> 1. Epoch 기반 스케줄러\n",
    ">> * StepLR, MultiStepLR, ExponentialLR 등\n",
    ">> * Epoch의 끝에서 학습률 변경\n",
    "> 2. Iteration 기반 스케줄러\n",
    ">> * OneCycleLR, CosineAnnealingWarmRestarts 등\n",
    ">> * 매 iteration 또는 mini-batch 처리 후 학습률 조정\n",
    "> 3. 적응형 스케줄러\n",
    ">> * 모델의 성능이나 메트릭을 기반으로 학습률 조정\n",
    ">> * ReduceLROnPlateau(검증 데이터셋의 성능 향상이 멈출 때 학습률 감소)\n",
    "> * 본 모델은 CosineAnnealingWarmRestarts을 사용하였으므로, iteration 안에서 스케줄러 조정.\n",
    "\n",
    "<br>\n",
    "\n",
    "### C. Torch 2.1 사용.\n",
    "> * `torch.compile(model)`\n",
    "> * `torch._dynamo.config.cache_size_limit = 512`\n",
    "\n",
    "<br>\n",
    "\n",
    "### D. Annotation 관련 주의 사항\n",
    "> * bbox가 그림 외곽에 생성되는 경우가 있음.\n",
    "> * bbox 생성 시, (x1, y1, x2, y2)가 (x_min, y_min, x_max, y_max)가 아닌 경우가 있음 - 반대로 Annotation한 경우.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.3333333333333333"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "4000/12000"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dev_e2",
   "language": "python",
   "name": "dev_e2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
