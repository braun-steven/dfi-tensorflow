# Deep Feature Interpolation
## Setup

### Python environment
Make sure you have `virtualenv` installed  
Run:
```bash
$ virtualenv -p /usr/bin/python3 env         # Create virtual python environment
$ ./env/bin/pip install -r requirements.txt  # Install all necessary requirements
```

### GPU
Follow [these instructions](http://www.computervisionbytecnalia.com/es/2016/06/deep-learning-development-setup-for-ubuntu-16-04-xenial/) up to point 4 to install CUDA on your system.
Use the gpu enabled tensorflow library by installing
```bash
$ ./env/bin/pip install tensorflow-gpu
```