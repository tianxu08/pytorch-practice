## Style Transfer

This is a practice of style transfer.
The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) and [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).

## Install dependencies

NOTE: I use anaconda in the experiement. It's easier to manage the package version in different environments. Here is an example to create environment
```Bash
# create new environment 
conda create --name style python=3.7
conda activate style
# install dependencies
pip install -r requirements.txt
```

## Data 

Under `data/`, run 
```Bash
./prepare_dataset.sh
```
Bascially, it will download the dataset from [coco](http://images.cocodataset.org/zips/train2014.zip) and then save the pictures under `data/coco`

## Run
### Start Visdom
Start the `visdom` server
```Bash
nohup python -m visdom.server &
```

### Train
```bash
python main.py train --use-gpu --data-root=data --batch-size=2
```

To train more styles, try different style images by `--style-path=mystyle.png`


