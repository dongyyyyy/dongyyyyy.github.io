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
  - 두가지 모듈인 **"cascade corner pooling"** 과 **"center pooling"** 을 만들었고, 이는 상단-좌측과 하단-우측 코너로부터 일치하는 정보를 풍부화하는 역할을하며, 중심지역에서 각각 더 인식 가능한 정보를 제공한다.

---

## Introduction

- In the current era, one of the most popular flowcharts is anchor-based, which placed a set of rectangles with pre-defined sizes, and regressed them to the desired place with the help of ground-truth objects.
  - 현재 시대에 가장 널리 사용되는 flowchars중 하나는 앵커 기반이며, 이는 사전에 정의된 크기의 사각형 세트를 배치하고 실측 객체의 도움으로 원하는 위치로 회귀합니다.


- **<span style="color:red">These approaches often need a large number of anchors to ensure a sufficiently high IoU(intersection over union) rate with the ground-truth objects, and the size and aspect ratio of each anchor box need to be manually designed</span>.**
  - 이러한 접근법들은 실측 객체와의 충분히 높은 IoU(교집합 / 합집합)보장하기 위하여 많은 수의 앵커들이 필요하고, 각 앵커박스의 크기와 종횡비는 수동적으로 설계되어져야합니다.


- **<span style="color:red">In addition, anchors are usually not aligned with the ground-truth boxes, which is not conducive to the bounding box classification task.</span>.**
  - 또한 앵커는 일반적으로 실측상자와 정렬되지 않기 때문에 바운딩 박스 분류 업무에 도움이 되지 않습니다.


- To overcome the drawbacks of anchor-based approahces, a keypoint-based object detection pipline named CornerNet was proposed.


- It represented each object by a parir of corner keypoints, which bypassed the need of anchor boxes and achieved the state-of-the-art one-stage object detection accuracy.


- Nevertheless, the performance of CornerNet is still restricted by its relatively weak ability of referring to the global information of an object.


- That is to say, since each object is constructed by a pair of corners, the algorithm is sensitive to detect the boundary of objects, meanwhile not being aware of which pair of keypoints should be grouped into obects.


 ![_config.yml](https://dongyyyyy.github.io/images/centerNet_fig1.JPG)
 - Consequently, as shown in Figure 1, it often generates some incorrect bounding boxes, most of which could be easily filtered out with complementary information, e.g., the aspect ratio.


 - The address this issue, we equip CornerNet with an ability of perceiving the visual patterns within each proposed region, so that it can identify the correctness of each bounding box by itself.


 - In this paper, we present a low-cost yet effective solution named **CenterNet,** which explores the central part of a proposal, i.e., the region that is close to the geometric center, with one extra keypoint.


 - Our intuition is that, if a predicted bounding box has a high IoU with the ground-truth box, then the probability that the center keypoint in its central region is predicted as the same class is hihgh, and vice versa.


 - Thus, during inference, after a proposal is generated as a pair of corner keypoints, we determine if the proposal is indeed an object by checking if there is a center keypoint of the same class falling within its central region. The idea, as shown in Figure 1, is to use  a triplet, instead of a pair, of keypoints to represent each object.



 - Accordingly, for better detecting center keypoints and corners, we propose two strategies to enrich center and corner information, respectively.

 - **cener pooling** : It is used in the branch for predicting center keypoints. Center pooling helps the center keypoints obtain more recognizable visual patterns within objects, which makes it easier to perceive the central part of a proposal. 
