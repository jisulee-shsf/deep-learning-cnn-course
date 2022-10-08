####
## Deep learning / CNN intermediate
####
#### ► [01_dl_keras_framework_sequential_api_v1_220629]
- Fashion MNIST dataset을 이용해, 10개의 의류 class에 속할 확률을 예측하는 모델 생성
- train / test x의 픽셀값 범위를 0-1 사이 값으로 scaling하여 예측 성능 변화 확인
- label의 one-hot encoding 여부에 따라, categorical crossentropy loss function 적용 실습
- label 값이 10개의 class 중에 속할 확률을 나타냄에 따라, 총합이 1인 softmax activation function 적용 실습
- 한 장의 2차원 배열 이미지의 경우, expand_dims()를 통해 3차원 배열로 변환 후 fit / predict() 진행 필요 확인
####
#### ► [02_dl_keras_framework_functional_api_v2_220701]
- dense layer를 기반으로 functional API를 활용한 모델 생성 실습
- model.summary() 출력 시, bias가 포함된 parameter 수의 이해와 계산 실습
####  
#### ► [03_dl_cnn_baseline_model_v3_220704]
- CIFAR10 dataset을 이용해, 10개의 object class에 속할 확률을 예측하는 모델 생성
- conv 연산을 연달아 적용한 후, max pooling을 배치하는 방향으로 model 생성
- conv 적용 시, batch size를 포함한 4차원 배열의 이미지 array만 입력 가능한 점 확인
- overfitting의 이해와 batch 단위의 랜덤 비율 노드를 drop하는 dropout의 이해와 적용 실습
- ResNet 이후 feature 손실 이슈로 pooling이 아닌 stride로 feature map의 크기를 줄이는 경향 확인
####
#### ► [04_dl_cnn_batch_normalization_v4_220704]
- batch normalization 적용 및 예측 성능 실습
- input이 layer를 통과할 때마다 분포가 변경되는 현상(internal covariate shift)을 방지하기 위함
- conv > batch normalization > activation 순서 배치
- ✓ 실습 결과, 기존 2-3 실습에 비해 test accuracy가 약 0.8030에서 0.8287로 향상
####
#### ► [05_dl_cnn_batch_size_v5_220705]
- batch size에 따른 예측 성능 실습
- batch size가 작은 경우, 상대적으로 자주 weight를 업데이트하여 정확한 최적화가 가능하다는 논문 내용에 기반
- ✓ 실습 결과, batch size가 작은 경우 비교적 예측 성능이 향상됨과 동시에 변동 안정성이 높음을 확인
####
#### ► [06_dl_cnn_callback_v6_220705]
- callback function 적용 실습
- 학습 interation 시, 등록한 callback API에 따라 특정 이벤트 발생 시 learning rate 등의 동적 작업 수행
- 본 실습에서 생성 및 등록한 callback API 조건은 하기와 같음
- ModelCheckpoint : val_loss가 가장 낮아지는 시점의 weight를 업데이트하여 별도 저장
- ReduceLROnPlateau : patience 기준, val_loss가 낮아지지 않을 경우 learning rate 갱신
- EarlyStopping : patience 기준, val_loss가 낮아지지 않을 경우 학습 중단
####
#### ► [07_dl_cnn_global_average_pooling_v7_220706]
- global average pooling 적용 실습
- feature map의 특정 영역을 sub sampling이 아닌 채널별 평균 값을 추출함
- feature map 채널 수가 많을 경우 global average pooling을 적용하나, 적을 경우 flatten 적용이 유리함
- 3차원 feature map과 1차원 classifier 연결 시, node와 parameter 수를 감소시켜 overfitting 방지
- node 수 : flatten(8,192개) / global average pooling(512개)
- parameter 수 : flatten(2,457,900개) / global average pooling(153,900개)
####
#### ► [08_dl_cnn_data_augmentation_v8_220707]
- data augmentation 적용 및 예측 성능 실습
- data를 늘리기 어려운 이미지의 경우, augumentation을 통해 다양한 학습 data 양을 늘려 overfitting 방지
- processing pipeline이 제공되나 랜덤 적용으로 변환 확률을 설정할 수 없음
- validation / test가 아닌 train 데이터에 한해 적용하는 점 확인
- ✓ 실습 결과, 기존 2-7 실습에 비해 test accuracy가 약 0.8619에서 0.8949로 향상
####
#### ► [09/10_cnn_transfer_learning_vgg16/xception_v9/10 220707/220708]
- Imagenet dataset으로 훈련된 pretrained model을 활용해 transfer learning 적용 및 예측 성능 실습
- include_top false 개념 이해와 custom한 classifier 설정 방법 실습
- feature extractor(CNN, activation, pooling) + classifier(fully connected layer, output layer)
- input size를 32x32에서 64x64로 변경하는 방법 실습
- CNN 실습을 통해 습득한 방법을 일괄 함수화하여 정리하는 방법 실습
- ✓ 실습 결과, Xception model을 통한 transfer learning 진행 시, 0.9308까지 test accuracy 향상
##  
#### ► [11_dl_cnn_intermediate_comprehensive_practice_220715]
- stanford dogs dataset을 이용해, 120여종의 품종을 구분하는 모델 생성 실습
####
