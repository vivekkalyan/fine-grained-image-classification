from trainer import *
from model import *
from utils import *
from adamw import *

from fastai2.vision import *

set_deterministic()

path ="data"
data = ImageDataBunch.from_csv(path, ds_tfms=get_transforms(), folder='cars_train', csv_labels='train.csv',  label_col=5, valid_pct=0, size=224, num_workers=8, bs=32).normalize(imagenet_stats)
test_data = ImageDataBunch.from_csv(path, folder='cars_test', csv_labels='test.csv', label_col=5, valid_pct=0, size=224, num_workers=8, bs=32).normalize(imagenet_stats)

train_loader = data.train_dl
valid_loader = data.valid_dl
test_loader = test_data.train_dl

model = Resnet('resnet50', data.c)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW([{"params":g.parameters()} for g in model.groups], lr=1e-2, weight_decay=1e-3, betas=(0.9, 0.99))
train = Trainer(model, criterion, optimizer, train_loader, None, div_lr=3)
train.train_one_cycle(20)
train.model.unfreeze()
train.train_one_cycle(40, lr=3e-3)
train.save_checkpoint('final_model')

train.test(test_loader)
