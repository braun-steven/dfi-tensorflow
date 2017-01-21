# Deep Feature Interpolation
Implementation of the Deep Feature Interpolation for Image Content Changes [paper](https://arxiv.org/abs/1611.05507) in tensorflow.

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

## Usage
```
./env/bin/python src/main.py -h
usage: Deep Feature Interpolation [-h] [--data_dir DATA_DIR]
                                  [--model_path MODEL_PATH] [--gpu]
                                  [--num_layers NUM_LAYERS]
                                  [--feature FEATURE]
                                  [--person_index PERSON_INDEX]
                                  [--list_features] [--tf]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -d DATA_DIR
                        Path to data directory containing the images
  --model_path MODEL_PATH, -m MODEL_PATH
                        Path to the model file (*.npy)
  --gpu, -g             Enable gpu computing
  --num_layers NUM_LAYERS, -n NUM_LAYERS
                        Number of layers. One of {1,2,3}
  --feature FEATURE, -f FEATURE
                        Name of the Feature.
  --person_index PERSON_INDEX, -p PERSON_INDEX
                        Index of the start image.
  --list_features, -l   List all available features.
  --tf                  Use Tensorflow for the optimization step
```
