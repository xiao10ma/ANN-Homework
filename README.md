# Facial Expression Recognition

## Installation
```bash
git clone https://github.com/xiao10ma/ANN-hm.git
cd ANN-hm
pip install -r requirements.txt
```

## How to run?

Move the data in to the data directory, it looks like this:
```bash
data
├── test
│   ├── Angry
│   ├── Happy
│   ├── Neutral
│   ├── Sad
│   └── Surprise
└── train
    ├── Angry
    ├── Happy
    ├── Neutral
    ├── Sad
    └── Surprise
```
If you are my teaching assistant, you need to copy the 'trained_model' directory from the files I provided into the project directory. The directory structure is as follows:
```bash
.
├── data
│   ├── test
│   └── train
├── face_dataset.py
├── model.py
├── net_utils.py
├── output
│   ├── AlexNet
│   ├── ResNet
│   └── VGG
├── README.md
├── requirements.txt
├── trained_model
│   ├── AlexNet
│   ├── ResNet
│   └── VGG
└── train.py
```
### Train
Then, you can run the project with just(default use AlexNet):
```
python train.py
```

You can choose different model(AlexNet, VGG, ResNet) in the main function. After that you need to change the record and model path of the args:
1. AlexNet:
```python
parser.add_argument('--record_path', default='./output/AlexNet/AlexNet-lr_{}epoch_{}'.format(LR, EPOCH), type=str)
parser.add_argument('--model_path', default='./trained_model/AlexNet', type=str)

network = AlexNet().to(device)
```

2. VGG:
```python
parser.add_argument('--record_path', default='./output/VGG/VGG-lr_{}epoch_{}'.format(LR, EPOCH), type=str)
parser.add_argument('--model_path', default='./trained_model/VGG', type=str)

network = VGG().to(device)
```

3. ResNet:
```python
parser.add_argument('--record_path', default='./output/ResNet/ResNet-lr_{}epoch_{}'.format(LR, EPOCH), type=str)
parser.add_argument('--model_path', default='./trained_model/ResNet', type=str)

network = ResNet50().to(device)
```

To visualize the training process, you can use tensorboard:
```bash
tensorboard --logdir={record_path}
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --data_source / -s
  Path to the data source directory face data set.
  #### --random / -m
  Flag to shuffle dataset.
  #### --model_path / -m 
  Path where the trained model should be stored (```trained_model/{Modelname}``` by default).
  #### --record_path
  Path to the record, you can use tensorboard to visualize it.
  #### --save_ep
  Every save_ep epochs, the program will save the trained model. Default 50.
  #### --save_latest_ep
  Every save_latest_ep epochs, the program will save the trained model. Default 10.

</details>
<br>

### Evaluate

I have implemented the evaluation function in train.py; you can call it directly.

---
If you have any questions, please contact me through email. My email: mazp@mail2.sysu.edu.cn
