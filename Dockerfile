# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir

COPY data ./data
COPY models ./models
COPY yolov8 ./yolov8
COPY config.yaml definitions.py video_object_detection.py ./


ENTRYPOINT [ "python" ]

CMD ["video_object_detection.py"]

