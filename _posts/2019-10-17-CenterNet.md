---
layout: post
title: Study Object Detection
date: 2019-08-21 17:58
summary: Summary Object Detection One-Stage & Two-Stage
categories: jekyll pixyll
---
# Object Detection 정리
---

## Abstract

- In object detection, keypoint-based approaches often suffer a large number of incorrect object bounding boxes, arguably due to the lack of an additional look into the cropped regions.
  -  객체 검출에 있어서, 키포인트기반의 접근법들은 많은 수의 일치하지 않는 객체 바운딩 박스를 제공하는데 이는 잘린 지역들에서의 추가적인 정보의 부족 때문이다.

- This paper presents an efficient solution which explores the visual patterns within each cropped region with minimal costs.
  - 이 논문은 최소한의 비용으로 각 잘려진 지역안에서의 시각적 패턴의 탐색하는 효율적인 해결책을 제시한다.

- Our approach, named CenterNet, detects each object as a triplet, rather than a pair,of keypoints, which improves both precision and recall.
  - 우리의 접근법인 CenterNet은 객체의 키포인트의 한 쌍이 아닌 세 쌍을 검출함으로써 precision과 recall 둘 다 향상시킨다.

- We designed two customized modules named "cascade corner pooling" and "center pooling", which play the roles of enriching information collected by both top-left and bottom-right corners and providing more recognizable information at the central regions, respectively.
  - 두가지 모듈인 "cascade corner pooling"과 "center pooling"을 만들었고, 이는 상단-좌측과 하단-우측 코너로부터 일치하는 정보를 풍부화하는 역할을하며, 중심지역에서 각각 더 인식 가능한 정보를 제공한다.


## Introduction
