# MTCNN++

pytorch implementation of **inference and training stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878) and [MTCNN++: A CNN-based face detection algorithm inspired by MTCNN](https://link.springer.com/article/10.1007/s00371-023-02822-0)

## Why this projects

There isn't an official implementation of MTCNN++, so we create this project forked from [mtcnn](https://github.com/BrightXiaoHan/FaceDetector) and adapted these features:

- Modify PNet, RNet and ONet as described in the MTCNN++ article.
- Removed facial landmarks detection.

## Installation

### Create virtual env (recommend)

```
python -m venv .venv
source .venv/bin/activate
```

### Installation dependency package

```bash
pip install  \-r requirements.txt
```

If you have gpu on your mechine, you can follow the [official instruction](https://pytorch.org/) and install pytorch gpu version.

### Compile the cython code
Compile with gpu support
```bash
python setup.py build_ext --inplace
```
Compile with cpu only
```bash
python setup.py build_ext --inplace --disable_gpu 
```

### Also, you can install mtcnn as a package
```
python setup.py install
```

## Test the code by example

We assume all these command running in the $SOURCE_ROOT directory.

#### Detect on example picture

```bash
python -m unittest tests.test_detection.TestDetection.test_detection
```

#### Detect on video

```bash
python scripts/detect_on_video.py --video_path ./tests/asset/video/school.avi --device cuda:0 --minsize 24
```

you can set device to 'cpu' if you have no valid gpu on your machine

## Basic Usage

```python
import cv2
import mtcnn

# First we create pnet, rnet, onet, and load weights from caffe model.
pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')

# Then we create a detector
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

# Then we can detect faces from image
img = 'tests/asset/images/office5.jpg'
boxes = detector.detect(img)

# Then we draw bounding boxes and landmarks on image
image = cv2.imread(img)
image = mtcnn.utils.draw.draw_boxes2(image, boxes)

# Show the result
cv2.imshwow("Detected image.", image)
cv2.waitKey(0)
```

## Doc
[Train your own model from scratch](./doc/TRAIN.md)

## Tutorial

[Detect step by step](./tutorial/detect_step_by_step.ipynb).

[face_alignment step by step](./tutorial/face_align.ipynb)
