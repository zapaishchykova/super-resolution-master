import os
import matplotlib.pyplot as plt

from data import DIV2K
from train import WdsrTrainer
from model.wdsr import wdsr_b

# Number of residual blocks
depth = 16#32

# Super-resolution factor
scale = 2

# Downgrade operator
downgrade = 'bicubic'

# Location of model weights (needed for demo)
weights_dir = f'weights/wdsr-b-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=1, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

trainer = WdsrTrainer(model=wdsr_b(scale=scale, num_res_blocks=depth),
                      checkpoint_dir=f'.ckpt/wdsr-b-{depth}-x{scale}')

# Train WDSR B model for 300,000 steps and evaluate model
# every 1000 steps on the first 10 images of the DIV2K
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
trainer.train(train_ds,
              valid_ds.take(10),
              steps=300,
              evaluate_every=10,
              save_best_only=False)

# Restore from checkpoint with highest PSNR
trainer.restore()

# Evaluate model on full validation set
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}')

# Save weights to separate location (needed for demo)
trainer.model.save_weights(weights_file)