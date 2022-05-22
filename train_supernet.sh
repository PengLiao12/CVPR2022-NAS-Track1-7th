CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_supernet.py run \
  --backbone resnet48 \
  --max_epoch 120 \
  --batch_size 128 \
  --lr 0.001 \
  --warmup 5 \
  --dyna_batch_size 4 \
  --pretrained checkpoints/resnet48.pdparams \
  --save_dir checkpoints/res48-depth \
  --log_freq 1 \
