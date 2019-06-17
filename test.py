from trainer import *
from model import *
from utils import *
from adamw import *

from fastai.vision import *

set_deterministic()

# new data here
path ="data"
data = ImageDataBunch.from_csv(path, ds_tfms=get_transforms(),
        folder='cars_test', csv_labels='test.csv',  label_col=5, valid_pct=0,
        size=299, num_workers=8, bs=16).normalize(imagenet_stats)

test_loader = data.train_dl

model = Resnet('resnext50', data.c)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW([{"params":g.parameters()} for g in model.groups], lr=1e-2, weight_decay=1e-3, betas=(0.9, 0.99))

# Train frozen model
train = Trainer(model, criterion, optimizer, None, None, div_lr=3,
        experiments_dir=".", save_dir=".")
train.load_checkpoint("best_model") #load model here
train.test(test_loader)
