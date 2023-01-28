# wustlcarla Docker

### Prerequisites

1. Install `docker`
2. Install `nvidia-docker2`
3. Add user to docker group

### Build Docker

```
sudo docker build -t wustlcarla/v1 .
```

### Use Built docker

Simple command:
```
sudo nvidia-docker run -it --rm --user root --net host -v $(pwd)/:/wustlcarla wustlcarla/v1 /bin/bash
```

Server ready command:
```
sudo nvidia-docker run -it --rm --user root --net host -v $(pwd)/:/wustlcarla -e NB_UID=$(id -u) -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e NB_GID=$(id -g) -e GRANT_SUDO=yes wustlcarla/v1 /bin/bash
```

Alternatively use (in root directory):

```
sh run_docker.sh
```

Note:
- This docker is Carla ready
- This docker is Jupyter Notebook ready
- Docker allows pygame visualization on servers with VNC/desktop mode enabled
