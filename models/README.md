# models/

Only needed for the MediaPipe vs OpenPose comparison (`python main.py compare`).
The OpenPose model is about 200 MB, so it is not stored in this repository. Download it into `models/openpose/`:

```
mkdir models\openpose
cd models\openpose
curl -L -o pose_deploy_linevec.prototxt https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
curl -L -o pose_iter_440000.caffemodel http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
```

OpenPose is run through OpenCV's DNN module, so no OpenPose build is required.
