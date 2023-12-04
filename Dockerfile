FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN apt update && apt install -y sudo

# installing required packages
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get install wget -y

ENV WANDB_API_KEY=8e9008b623a334edf472f175d059c25c9aa66207
