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
  본인이 학습시킬 cfg파일의 max_batches값을 변경해주어야 한다. ( 해당 학습 데이터는 총 500200의 학습을 이미 했기 때문에 수정 안하고 학습을 진행하면 바로 학습이 종료된다. )  
  ![_config.yml](https://dongyyyyy.github.io/images/cfg.JPG)  
    * 기존 max_batches값을 600200으로 수정하여 학습하기 위한 방법    
    * ```./darknet detector train [data파일명] [cfg파일명] [이전에 학습된 weights 파일명 ] ``` 으로 학습 가능  


---
##### 소스 부분  
* 학습 부분에서 소스를 봐야 되는 클래스는 *examples/darknet.c , detector.c* , *src/network.c , option_list.c , list.c , utils.c , parser.c , convolutional_layer.c , detector.c , image.c , matrix.c , image_opencv.cpp* 와 같은 파일들을 확인하면 좋습니다.
* 소스 흐름도 ( train부분과 detection부분 )은 [StarUML5.0](https://dongyyyyy.github.io/information/darknet.uml)을 통해서 확인할 수 있습니다.  

###### 여기서는 학습에서 가장 중요한 Anchor Boxes에 관련된 내용을 다룰 예정  
* 소스 부분에서 [yolo_layer.c](https://github.com/dongyyyyy/darknet/blob/master/src/yolo_layer.c)클래스의 forward_yolo_layer함수를 확인하시면 좋습니다.
