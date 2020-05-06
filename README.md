# Hiragana-Recognition
The project is aimed to the Neural Network course project, which compares the state-of-the-art image classification networks. 

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
    * --vgg: Use VGG 19 with Batch Normalization.
    * --inception: Use Inception V3 Net.
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
