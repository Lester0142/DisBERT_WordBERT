# DisBERT_WordBERT

This repository contains the Python script for training a metaphor identification model that leverages BERT. The model is trained using a dataset in `.tsv` format, containing sentences with corresponding labels for word-level metaphor identification. This script is designed to train and fine-tune the model and evaluate its performance using F1-score, Precision, and Recall metrics.

## Prerequisites

Before running the script, ensure that you have Python 3.7.6 and Nvidia CUDA installed on your machine.

Python 3.7.6: https://www.python.org/downloads/release/python-376/ \
CUDA: https://developer.nvidia.com/cuda-downloads

Also these are some dependencies required for Ubuntu System to pip install pillow:
#### Debian-based systems
```
sudo apt-get update
sudo apt-get install libtiff4-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.5-dev tk8.5-dev python-tk
```

To ensure the appropriate requirements are successfully installed. Run the following in your terminal. The terminal respond should be similar as shown in the block below.
```
python -V
>>> Python 3.7.6

nvcc --version
>>> nvcc: NVIDIA (R) Cuda compiler driver
>>> Copyright (c) 2005-2021 NVIDIA Corporation
>>> Built on Thu_Nov_18_09:45:30_PST_2021
>>> Cuda compilation tools, release 11.5, V11.5.119
>>> Build cuda_11.5.r11.5/compiler.30672275_0
```


## Run the Script

#### 1. Clone the repository locally
```
git clone git@github.com:Lester0142/DisBERT_WordBERT.git
```

#### 2. Set up and activate Environment
```
cd DisBERT_WordBERT

python -m venv venv

linux:
    source venv/bin/activate
windows:
    .\venv\Scripts\activate
```

#### 3. Install required depencies and packages
```
pip install -r requirements.txt

python

import nltk

nltk.download('wordnet')
```
close python interpreter using ctrl + 'd'

#### 4. Amend the config file as required
```
config file can be found in ./src/config/main_config.cfg
```

#### 5. Run the script
```
python ./src/main.py
```

