---
layout: post
title: Study Object Detection
date: 2019-08-21 17:58
summary: Summary Object Detection One-Stage & Two-Stage
categories: jekyll pixyll
---
# Object Detection 정리
---

## 요약
* Object Detection에 사용되는 Architecture들을 살펴보며 어떤식으로 객체를 검출하는지에 대해서 다룰 예정


* 가장 많이 사용하는 검출 방법으로는 One-Stage & Two-Stage 방식이 존재함.  


* One-Stage는 속도면에서 높은 성능을 얻을 수 있으며 Two-Stage경우에는 속도보다는   정확성면에 높은 비중을 두고 학습을 함.


---

## Two-Stage Detectors
* Two-stage detectors는 2개의 구역으로 나눌 수 있다.
  1. Proposal generation 구역 생성
  2. making predictions for these proposals 해당 구역을 통한 예측

### R-CNN
* R-CNN은 two-stage object detection의 기반을 다진 구조이다.


* 과거의 object detection 방식(SegDPM)의 경우 Pascal VOC2010에서 40.4% mAP(mean Average Precision)을 얻은 반면 R-CNN의 경우에는 53.7% mAP를 획득하였다.


* R-CNN은 다음과 같이 나눠진다.
  1. Proposal generation 제안 생성
  2. Feature extraction 특징 추출
  3. Region Classification 구역 분류


1. Proposal generation
  * Class는 신경쓰지 않고 이미지에서 region을 추출하는 작업.
  * 해당 R-CNN에서는 SS(Selective Search)방법을 사용하여 해결.
  * SS는 비슷한 질감이나 색,감도를 가진 인접한 픽셀들을 연결하여 Bounding Box를 생성
  * 이를 통해 나온 Bounding Box를 1개씩 CNN에 입력.

2. Feature extraction
  * 1단계를 통해 proposal 된 region을 CNN의 input값으로 넣어 feature extract를 진행.
  * R-CNN에서는 AlexNet, VggNet등을 모델로 사용함.

3. Region Classification
  * 2단계를 통해 도출된 Feature map을 사용하여 객체여부 판별 및 객체 분류를 한다.
  * R-CNN에서는 SVM (Support Vector Machine)을 활용하여 객체 분류를 함.

* 정리
  * 결과론적으로 기존에 존재한 Object detection들 보다 높은 성능을 이끌어 내었다.
  * 하지만 속도면에 있어서 너무 느린 점이 존재. ( Region proposal을 통해서 2000개의 region을 생성한다면 총 CNN을 2000번 학습시켜야함. )


---

### Fast R-CNN
* 기존의 R-CNN의 느린 속도를 보완하고자 등장

* 기존에는 Region Proposal이 CNN 이전에 시행되어 총 2000번의 Image를 CNN 작업을 따로 했지만, Fast R-CNN은 Region Proposal 작업을 CNN에서 처리함으로써 연산량을 급격히 줄임.

* Region을 생성한 후 모든 다른 크기의 Region을 Classfication을 하기 위해서는 각 Region의 이미지크기를 통일시켜야함. 이 부분을 R-CNN의 경우에는 warp을 활용하여 해결했지만 Fast R-CNN은 Spatial Pyramid Pooling을 사용함.

* Fast R-CNN은 SPP layer의 single level pyramid만 사용하여 이를 ROI pooling layer이라함.

* R-CNN에서는 softmax classifier 와 linear bounding box regressor을 따로 학습.
* Fast R-CNN은 두 함수의 loss를 더한 multi-task loss기반으로 두가지를 동시에 학습.

---

### Faster R-CNN
