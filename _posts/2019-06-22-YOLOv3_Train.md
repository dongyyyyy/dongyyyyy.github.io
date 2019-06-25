---
layout: post
title: YOLOv3-Train
date: 2019-06-22 16:39
summary: YOLOv3 darknet 소스 부분 중 직접 이미지 학습 시 소스 동작 요약
categories: jekyll pixyll
---
# YOLOv3-Train
---
* 학습시 입력 값  
  * 기본 weight파일의 경우에는  [[weight파일](http://pjreddie.com/media/files/darknet53.conv.74)]에서 설치 가능  
    * ``./darknet detector train [data파일명] [cfg파일명] [weight파일명] ``  
  * 만약 coco데이터와 동일하게 학습할 경우 (80개의 객체에 대해서 학습)  
      *   ``./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74``
  * 기존 weights(YOLO에서 주어준 학습 데이터)를 사용하여 transfer learning을 하기 위해선
  본인이 학습시킬 cfg파일의 max_batch값을 변경해주어야 한다. ( 해당 학습 데이터는 총 500200의 학습을 이미 했기 때문에 수정 안하고 학습을 진행하면 바로 학습이 종료된다. )    ![_config.yml](https://dongyyyyy.github.io/images/cfg.JPG)
