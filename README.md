# MeshVertexNet in PyTorch



# Getting Started 

### Installation
- Clone this repo:
``` bash 
git clone https://github.com/JsBlueCat/MeshVertexNet.git
cd MeshVertexNet
```
- Install dependencies: [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and [docker](https://docs.docker.com/get-started/)

- First intall docker image

```bash
cd docker
docker build -t your/docker:meshvertex .
```

- then run docker image
```bash
docker run --rm -it --runtime=nvidia --shm-size 16G -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /your/path/to/MeshVertexNet/:/meshvertex  your/docker:meshvertex bash
```


### 3D Shape Classification on ModelNet40

```bash 
cd /meshvertex
sh script/modelnet40/train.sh 
```
