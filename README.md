# You Only Look Once

1.Train YOLOv2 object detector from scratch with Tensorflow.

## Usage
Prepare two files: 

data.csv (three columns: filenames, rois, classes - each row contains image filepath, list of rois (each [x,y,w,h]), list of classes) and anchors.txt (each row contains width and height of one anchor).
```
python make_tfrecord.py
python train.py
python eval.py
```

<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/neta.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/netb.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/netc.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/loss2.png" width="300">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/merge.png" width="700">

2.Evaluate YOLOv2 model trained with COCO dataset using Tensorflow. Conversion from Darknet to Tensorflow framework done with darkflow project.

<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/a2.png" width="400">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/a4.png" width="290">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r1.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r2.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r3.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r4.png" width="700">
<img src="https://github.com/WojciechMormul/yolo2/blob/master/imgs/r5.png" width="700">

