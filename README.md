# You Only Look Once

1. Train YOLOv2 object detector from scratch with Tensorflow.

Joseph Redmon's, Ali Farhadi's paper: https://arxiv.org/pdf/1612.08242.pdf

## Usage
Prepare two files: 
data.csv (three columns: filenames, rois, classes - each row contains image filepath, list of rois (each [x,y,w,h]), list of classes) and anchors.txt (each row contains width and height of one anchor).
```
python make_tfrecord.py
python train.py
python eval.py
```

<img src="https://s13.postimg.org/rci0unzjb/neta.png" width="400">
<img src="https://s13.postimg.org/g05fcvyk7/netb.png" width="400">
<img src="https://s13.postimg.org/el3uo6cwn/netc.png" width="400">
<img src="https://s9.postimg.org/ossrlazcf/loss2.png" width="300">
<img src="https://s9.postimg.org/dgg63jlin/merge.png" width="700">


2. Evaluate YOLOv2 model trained with COCO dataset using Tensorflow. Conversion from Darknet to Tensorflow framework done with darkflow project.

<img src="https://s14.postimg.org/zfqjg9jzl/image.png" width="400">
<img src="https://s13.postimg.org/mb61fwpxz/image.png" width="290">
<img src="https://s13.postimg.org/dub4ilslj/image.png" width="700">
<img src="https://s13.postimg.org/jihf9i4nr/image.png" width="700">
<img src="https://s13.postimg.org/o4djhv5mf/image.png" width="700">
<img src="https://s13.postimg.org/sq9nq7yvb/image.png" width="700">
<img src="https://s13.postimg.org/uuu0rb87r/image.png" width="700">

