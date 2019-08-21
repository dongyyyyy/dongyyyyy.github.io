---
layout: post
title: Study Object Detection
date: 2019-08-21 17:58
summary: Summary Object Detection One-Stage & Two-Stage
categories: jekyll pixyll
---
# Object Detection 정리
---
* 해당 내용은 [[Recent Advances in Deep Learning for Object Detection](https://arxiv.org/abs/1908.03673)]논문을 참조하여 작성하였음.


* 해당 논문에는 각 Architecture에 대해서 자세히 나오지는 않았기 때문에 추가적으로 필요한 내용은 다른 논문 혹은 구글링을 통해서 작성하였음.

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
  3. region classification 구역 분류


* Proposal generation을 하기 위해서 SS(Selective Search)를 통하여 구역을 생성한다.
