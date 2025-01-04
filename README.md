# A2MADA-YOLO
This is the official implementation of [ A2MADA-YOLO: Attention Alignment Multiscale Adversarial Domain Adaptation YOLO for Insulator Defect Detection in Generalized Foggy Scenario] based on YOLOv7, v8 and v9.

## Installation
Please follow the instruction in [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) to install the requirements of YOLOv7.  
Please follow the instruction in [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) to install the requirements of YOLOv8.  
Please follow the instruction in [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) to install the requirements of YOLOv9.

## Training
Data preparation

* Download UPID dataset images ([heitorcfelix/public-insulator-datasets](https://github.com/heitorcfelix/public-insulator-datasets))
* Fogging the UPID dataset using the method in ([zhangzhengde0225/FINet](https://github.com/heitorcfelix/public-insulator-datasets))

``` shell
# train A2MADA-YOLOv7 models
python A2MADA-YOLOv7/train.py --batch-size 16 --data data/data.yaml --cfg cfg/training/yolov7.yaml --hyp data/hyp.scratch.p5.yaml

# train A2MADA-YOLOv8 models
python A2MADA-YOLOv8/main.py

# train A2MADA-YOLOv9 models
python A2MADA-YOLOv9/train.py --batch-size 16 --data data/data.yaml --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml
```

## Testing

``` shell
# test A2MADA-YOLOv7 models
python A2MADA-YOLOv7/test.py --data data/data.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights best.pt

# test A2MADA-YOLOv8 models
python A2MADA-YOLOv8/test.py

# test A2MADA-YOLOv9 models
python A2MADA-YOLOv9/val.py --data data/data.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights "best.pt" --task test
```
