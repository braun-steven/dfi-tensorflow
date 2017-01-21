# Deep Feature Interpolation
## Setup
### Model
Download the VGG19 model from [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

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
