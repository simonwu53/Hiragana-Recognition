# Hiragana-Recognition
The project is aimed to the Neural Network course project, 
which compares the state-of-the-art image classification networks for hand-written hiragana characters. 

The experiment will be divided into two parts: single character recognition and multi-characaters recognition. 
The second task will localize the characters in the image and identify them.

# Usage
1. Install all libraries required in "requirements.txt"
2. Change the training configuration in "config.py"
3. Use the following command to try the training process.

```shell script
python inference.py --train --vgg
```

* To continue training process, run the following command:
```shell script
python inference.py --train --vgg --load /path/to/checkpoint/file.tar
```

* Available models:
    * --vgg: Use VGG 19 with Batch Normalization. Minimum input size 224x224x3
    * --inception: Use Inception V3 Net. Minimum input size 299x299x3, batch size must > 1.
    * --simple: Use customized CNN.
    
* Available modes:
    * --train: Start training process.
    * --test: Start testing process.
    
* Available inputs:
    * --dataset: Path to the dataset.
    * --load: path to the saved 'tar' checkpoint file.

## Dependencies
Please refer to the "requirements.txt" file in the project folder.
1. torch~=1.4.0
2. numpy~=1.18.2
3. tqdm~=4.45.0
4. torchvision~=0.5.0
5. scikit-learn~=0.22.2.post1
6. matplotlib~=3.2.1
7. opencv-python~=4.2.0

## Results
| \ | VGG 19 with BN | Inception v3 | Customized CNN |
| :--- | :---: | :---: | :---: |
| Input Size | 224x224x3 | 299x299x3 | 50x50x1 |
| Normalized | True | True | True |
| Batch Size | 8 | 8 | 8 |
| Learning Rate | 1e-3 | 1e-3 | 1e-3 |
| Train/Test Split | 85%/15% | 85%/15% | 85%/15% |
| Epochs | 30 | 30 | 30 |
| Parameters | 126,001,031 | 24,543,342 | 1,712,103 |
| Validation Loss | 0.7290 | 0.2385 | 0.6475 |
| Validation Accuracy | 98.74% | 99.5% | 98.47% |
| Inference Time| 0.0294s/it | 0.0189s/it | 0.0020s/it |





| ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `VGG 19 with BN` | ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Inception v3` | ![#03fcfc](https://via.placeholder.com/15/03fcfc/000000?text=+) `Customized CNN` |
| :---: | :---: | :---: |

| Training Accuracy | Training Loss|
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/simonwu53/Hiragana-Recognition/master/results/comparison/Train_Accuracy.svg?token=AGQWTMQTDBJFE3UU2WMAV526XRKCC" width="350" alt="Train_Accuracy"> | <img src="https://raw.githubusercontent.com/simonwu53/Hiragana-Recognition/master/results/comparison/Train_Loss.svg?token=AGQWTMWACYWFYZBAOGFSEE26XRKH2" width="350" alt="Train_Loss"> |

| Validation Accuracy | Validation Loss|
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/simonwu53/Hiragana-Recognition/master/results/comparison/Validation_Accuracy.svg?token=AGQWTMQ34BKQIEAQ67RKC7S6XRKK4" width="350" alt="Validation_Accuracy"> | <img src="https://raw.githubusercontent.com/simonwu53/Hiragana-Recognition/master/results/comparison/Validation_Loss.svg?token=AGQWTMV43ZVFBHKZUCFKKGS6XRKMM" width="350" alt="Validation_Loss">|
