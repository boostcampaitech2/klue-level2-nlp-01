# KLUE-LEVEL2-NLP-01

**Run & Learn 팀 KLUE 대회 레퍼지토리**

# 대회 개요

![](https://i.imgur.com/rJTqT1N.png)

## 동기

- 문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다.
- 그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

## 목표

- 이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 
- 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 

## 데이터 샘플

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.

subject_entity: 썬 마이크로시스템즈

object_entity: 오라클
```

- relation: 단체:별칭 (org:alternate_names)

- input: sentence, subject_entity, object_entity의 정보를 입력으로 사용 합니다.

- output: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 제출해야 합니다! class별 확률의 순서는 주어진 dictionary의 순서에 맡게 일치시켜 주시기 바랍니다.


# 사용법

- main_cfg.yaml 을 설정한다.
- main.py을 실행하여 훈련시킨다.
- models 폴더에 생성된 모델의 결과물을 제출한다.
- ~~끗~~

# 소스트리
![](https://i.imgur.com/mNLXKsW.png)

- custom_models.py
    - 사용자 정의 모델을 생성합니다.
- data_load.py
    - Load 관련된 모듈을 모아놨습니다.
- EDA_AEDA.py
    - AEDA를 구현하였습니다.
- hyper_parameters.py
    - Optuna를 통한 하이퍼파라미터 최적화를 실행합니다.
- inference.py
    - 제작한 모델을 통해 대회 테스트 데이터를 평가합니다.
- pre_process.py
    - 데이터 전처리를 진행합니다.
- tokenizing.py
    - 토큰화를 진행합니다.
- train.py
    - 모델을 훈련시키는 메인 파일입니다.
- util.py
    - label_to_num 등 기타 유틸이 담겨있습니다.
- validation.py
    - 모델을 평가하는 함수입니다.

# 개발 과정

- EDA(데이터 분석) 진행
- 데이터의 특성을 파악하고 SUB, OBJ를 바꿔 훈련데이터 증가
- 코드 통일을 위한 코드포맷터(black) 추가
- Stratified K Fold 구현
- 폴더 구조 정리
- Hydra를 추가하여 Arguments 통일
- 데이터 전처리 과정 추가, 학습데이터의 SUB,OBJ 타입을 활용
- 커스텀 모델 추가
- AEDA 상속 및 파인튜닝
- 하이퍼 파라미터 최적화 구현

