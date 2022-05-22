

# CVPR2022-NAS-Track1-10th

This is the code to participate in CVPR2022 NAS Track1.

The official website of the competition is: [CVPR2022-NAS-Track1](https://aistudio.baidu.com/aistudio/competition/detail/149/0/leaderboard)

# Preparatory Work
./checkpoints Storage model results, provided by the official website or self-training

./data stores files containing 45,000 model codes, which are officially provided.
```
./checkpoints
./data
./hnas
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

```bash
nohup sh train_supernet.sh >  train.txt 2>&1 &
``` 

# Evaluate

```bash
nohup sh evaluate.sh >  train.txt 2>&1 &
``` 

Corresponding to the download address of the supernet file:  [model](https://pan.baidu.com/s/1qk0favYmEy25V5ZxFYflxQ )

code：nwwv