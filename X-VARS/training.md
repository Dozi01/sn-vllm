# Training

This README outlines the steps required for training. We train X-VARS on 22k video-question-answer triplets. The training on 2 A100 40GB GPUs takes about 2 hours.

## Download weights

Download the [base_model_videoChatGPT](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ) weights. We use the [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main?tab=readme-ov-file) multi-modal language model as our foundation.
Save the weights in "X-VARS/X-VARS".

## Download the SoccerNet-XFoul dataset

The annotations will be available soon! Stay tuned🔥

## Train X-VARS
```
accelerate launch --config_file "path/to/default_config.yaml" training.py
```