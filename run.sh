#!/bin/sh
rm -rf runs/
python3 -m pip install -r requirements.txt
python main.py --weights yolov7.pt --source "video2.mp4"