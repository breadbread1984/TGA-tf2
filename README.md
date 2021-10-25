# Temporal Group Attention

## Introduction

this project implements video super resolution algorithm TGA.

## prepare dataset

download VIMEO90k dataset [here](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip)

## how to train

train with command

```shell
python3 train.py --vimeo_path=<viemo 9k path>
```

## how to save model

save the trained model with the command

```shell
python3 train.py --save_model
```

