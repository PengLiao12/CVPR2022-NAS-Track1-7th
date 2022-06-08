

# CVPR2022-NAS-Track1-7th

This is the code to participate in CVPR2022 NAS Track1.

The official website of the competition is: [CVPR2022-NAS-Track1](https://aistudio.baidu.com/aistudio/competition/detail/149/0/leaderboard)

# Preparatory Work
./checkpoints Storage model results, provided by the official website or self-training.

./data stores files containing 45,000 model codes, which are officially provided.

./paddleslim/nas/ofa/resofa.py  add the function to activate the subnet according to the code.

The data set is [imageNet2012](https://image-net.org/).
```
./checkpoints
./data
./hnas
./paddleslim/nas/ofa/resofa.py
model.py
sampling.py
train_supernet.py
train_supernet.sh
evaluate.py
evaluate.sh
README.MD
requirements.txt
```

# Train

The training phase is based on the baseline code and generates a specified number of trained architectures in each epoch, that is, the number of images in the training set divided by the batch size.

See sampling.py for the training method of selecting the trained framework.

Divide the specified quantity into 7 parts with the fifth digit code. Then permutations and combinations of the first four codes (i.e., four SATGe) were selected as candidate pools, and further combinations satisfying this part of the number were selected by sampling. Finally, the rest of the code bits, the channel of the block, are generated randomly. See sampling.py for details.

```bash
nohup sh train_supernet.sh >  train.txt 2>&1 &
``` 

# Evaluate

```bash
nohup sh evaluate.sh >  train.txt 2>&1 &
``` 

Corresponding to the download address of the supernet file:  [model](https://pan.baidu.com/s/1qk0favYmEy25V5ZxFYflxQ )

code：nwwv
