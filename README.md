# Mutual Mean Teaching

Unofficial implementation of [Mutual Mean Teaching: Pseudo Label Refinery for Unsupervised Doamin Adaptation on Person Re-Identification, ICLR 2020](https://arxiv.org/abs/2001.01526)

## 환경

1. python3.6.9 기반의 venv 사용

```bash
python3 -m venv .venv   # venv 환경 생성
. .venv/bin/activate    # venv환경 활성화
pip3 install -r requirements.txt # 필요 패키지 다운로드
```

2. pytorch 1.8.1+cu111
3. torchvision 0.9.1+cu111
4. scikit-learn 0.24.2

## Preprocess Data

벤치마크 데이터셋 다운로드 및 전처리
Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)

```bash
python prepare.py
```

## Pretrain

ImageNet을 학습한 ResNet50을 활용하여 벤치마크 데이터 지도 학습

```bash
python train.py --pretrain
```

## Main Train

Mutual Mean Teaching 활용한 비지도 학습

```bash
python train.py
```

## Test

- pretrain 모델 성능 테스트
- Rank@1:0.415083, Rank@5:0.626188, Rank@10:0.710214, mAP:0.175307

```bash
python test.py --pretrain   # feature .mat 파일로 저장
python evaluate_gpu.py
```

- MMT 모델 성능 테스트

```bash
python test.py
python evaluate_gpu.py
```

### TODO

1. 하이퍼파라미터 세팅 config 파일 작성
2. validation loop
3. pretrain/MMT 성능 개선
