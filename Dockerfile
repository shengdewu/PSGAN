FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10.0

RUN apt-get -y update && apt-get install -y build-essential cmake &&  pip3 install requests dlib

COPY data_loaders /home/psgan/data_loaders
COPY ops /home/psgan/ops
COPY psgan /home/psgan/psgan
COPY concern /home/psgan/concern
COPY faceutils /home/psgan/faceutils
COPY tools /home/psgan/tools
COPY train.py /home/psgan
COPY setup.py /home/psgan
COPY dataloder.py /home/psgan

WORKDIR /home/psgan
ENTRYPOINT ["python3", "train.py"]