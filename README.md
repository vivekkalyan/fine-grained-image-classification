# fine-grained-image-classification

Fine grained image classification on the [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset. The model achieves ~89.7% accuracy on the test set.

# Model

To achieve a fair evaluation of the final model, the test set is not used at all at figuring out model architecture and hyper-parameters. They are chosen using a 20% validation set. Once they are decided, the model is trained again on the entire training+validation set and evaluated on the test set.

The model is trained on a frozen (except BN layers) resnet50 for 20 epochs and then finetuned with discriminative learning rates for 40 epochs. The final model uses 3e-4, 1e-4, 3e-3 as learning rates for each third of the model.

The [AdamW](https://arxiv.org/abs/1711.05101) implementation of Adam is used together with the [One Cycle Policy](https://arxiv.org/abs/1803.09820) as it is seen that training results are much better and converges faster.

The [fastai](https://github.com/fastai/fastai) library is used to do data augmentation.

# Installation

Clone repository

```
git clone https://github.com/vivekkalyan/fine-grained-image-classification.git
cd fine-grained-image-classification
```

Setup virtual environment

```
virtualenv -—python=python3 env
. env/bin/activate
```

Download and extract data

```
make
```

# Reproduce results

`run.py` file has the parameters used to train the final model that achieves ~89.7% accuracy in 60 epochs.

```
python run.py
```