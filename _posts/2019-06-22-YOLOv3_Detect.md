---
layout: post
title: YOLOv3-Detect
date: 2019-06-25 16:16
summary: YOLOv3 darknet 소스 부분 중 이미지 검출 시 소스 동작 원리 요약
categories: jekyll pixyll
---
# YOLOv3-Detect
---
##### 소스 부분  
````
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }

        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];
        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X); // 모델을 통하여 실제 예측
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms); // 겹치는 부분 제거 부분
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes); // 이미지 검출 부분 박스 그리는 함수
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
````
>* Detection 부분은 학습부분에 비해서 쉬운편에 속한다.
* 간단하게 설명하면 학습하려는 이미지에 대해서 Convolutional layer들을 통해서 feature map을 추출하고 추출한 feature map을 통하여 각 grid cell에서 3개의 Anchor Box를 통하여 iou를 계산한다.
* 계산하는 과정에서 일정 iou이상일 경우에 가장 큰 object score를 저장하고 나머지 score는 0으로 설정하여 없앤다.
* 모든 Grid cell을 통해서 각 Anchor Box마다 객체를 가지는지에 대해서 파악한 후 최종적으로 일정 값 이상으로 겹치는 부분이 있는 경우에 (하나의 이미지에 대해서 여러 Bounding Box가 겹칠 경우 중복을 없애주는 작업) 작은 값을 지우므로써 겹쳐지는 상황을 없애는 작업을 한다.
* 최종적으로 찾은 객체 값들의 정보를 사용자에게 줌으로써 detection 부분은 끝이 난다.
* 여기서는 하나의 이미지에 대해서 검출을 어떻게 하는지에 대한 부분에 대해서 소스 설명을 할 예정이다.
