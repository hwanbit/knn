# K-NN 분류 모델 구현 (Iris 데이터셋)

## 📌 프로젝트 목적
**K-최근접 이웃(K-Nearest Neighbors, K-NN) 알고리즘의 작동 원리**를 이해하기 위해 Scikit-learn 라이브러리의 구현체를 사용하지 않고, **Python으로 구현**한 코드입니다.

## 🎯 학습 목표
- K-NN 알고리즘의 핵심 개념 이해
- 유클리드 거리 계산 방법 습득
- 다수결 투표(Majority Voting) 메커니즘 이해
- 머신러닝 분류 문제의 기본 워크플로우 학습

## 🛠️ 주요 특징 및 구현 세부 사항

### 1. 의존성
- `scikit-learn`: Iris 데이터셋 로드
- `math`: 유클리드 거리 계산
- `collections.Counter`: 이웃 라벨의 빈도 계산

### 2. 데이터 준비
- **데이터셋**: Iris 데이터셋 (총 150개 샘플, 3개 클래스)
  - 클래스: Setosa, Versicolor, Virginica
- **특징(Features)**: 4개 (꽃받침 길이/너비, 꽃잎 길이/너비)
- **데이터 분할**:
  - 학습 데이터: 11번째 샘플부터 끝까지 (`X[10:]`, 140개)
  - 테스트 데이터: 앞 10개 샘플 (`X[:10]`, 10개)

### 3. K-NN 핵심 로직

#### 하이퍼파라미터
- **K 값**: 5 (가장 가까운 5개의 이웃 고려)

#### 알고리즘 단계
1. **거리 계산**: 테스트 샘플과 모든 학습 데이터 간의 유클리드 거리 계산
   
   $$d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

2. **이웃 선택**: 거리를 기준으로 정렬하여 가장 가까운 k개 이웃 선택

3. **예측**: k개 이웃의 라벨 중 최빈값을 최종 예측값으로 결정

### 4. 코드 구조
```
euclidean_distance(point1, point2)  # 두 점 사이의 유클리드 거리 계산
└─ 메인 분류 로직
   ├─ 거리 계산 (모든 학습 데이터)
   ├─ 정렬 및 k개 이웃 선택
   └─ 다수결로 예측
```

## 🖥️ 실행 방법

### 1. 환경 설정
```bash
!pip install scikit-learn
```

### 2. 실행
Jupyter Notebook 환경에서 `sun_knn.ipynb` 파일의 셀을 순서대로 실행합니다.

### 3. Google Colab에서 실행
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CfvTnaLENnP8ZOkaaJ2TyY_Xq92G_9Zd?usp=sharing)

## 📈 실행 결과

### 예측 결과
| 구분 | 결과 |
|------|------|
| 예측 라벨 | `['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']` |
| 실제 라벨 | `['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']` |

### 성능 지표
- **정확도(Accuracy)**: 100% (10/10)
- **참고**: 테스트 데이터가 모두 같은 클래스(setosa)이므로, 더 다양한 평가를 위해서는 전체 데이터셋에 대한 교차 검증이 필요합니다.
