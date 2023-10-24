#!/bin/bash
echo "hello world"
source /home/user/anaconda3/etc/profile.d/conda.sh
pwd
cd /home/user/yolo_ai/onnx-yolov8/
conda activate onnx2
conda info
python -V
python video_object_detection.py
read -p "Press enter to continue"
