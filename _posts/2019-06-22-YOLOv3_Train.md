---
layout: post
title: YOLOv3-Train
date: 2019-06-22 16:39
summary: YOLOv3 darknet 소스 부분 중 직접 이미지 학습 시 소스 동작 요약
categories: jekyll pixyll
---
# YOLOv3-Train
---
학습시 입력 값  
기본 weight파일의 경우에는  [[weight파일](http://pjreddie.com/media/files/darknet53.conv.74)]에서 설치 가능  
``./darknet detector train [data파일명] [cfg파일명] [weight파일명] ``  
만약 coco데이터와 동일하게 학습할 경우 (80개의 객체에 대해서 학습)  
``./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74``
