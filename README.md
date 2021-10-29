# Temporal Group Attention

## Introduction

this project implements video super resolution algorithm TGA.

## prepare dataset

download VIMEO90k dataset [here](http://data.csail.mit.edu/tofu/dataset/vimeo_settuplet.zip)

## how to train

train with command

```shell
python3 train.py --vimeo_path=<viemo 9k path>
```

a pretrained checkpoint can be download [here](https://pan.baidu.com/s/1kTF-IP_YjdrZZHkhlNeSNw), passcode is **qnqm**

## how to save model

save the trained model with the command

```shell
python3 train.py --save_model
```

## how to test saved model

test the model with the command

```shell
python3 test.py
```
