# Multi-modal Aerial View Object Classification Challenge (MAVOC 2022) Solution

This repo contains the solution for MAVOC challenge track 2 (SAR+EO), which utilized **data augmentation, focal loss, semi-supervised learning, label calibration** etc. techniques to tackle with the given task. Detailed information can be found in [method description](./_assets/mavoc22_method_description.pdf). This solution ranked *6th* in the final leaderboard in test phase.

### Dataset structures

The datasets should be organized as follows. The `/path/to/dataset` in codes and configs refers to the root of this structure.

```
dataset
    - train_images
        - 0
        - 1
        ...
        - 9
    - test_images
        - test_eo
        - test_sar
    - valid_images
        - valid_eo
        - valid_sar
```

### Training and Inference

Firstly, use the scripts in `./preprocess_scripts` to generate csv file for dataloaders, then pip install required libraries listed in `requirements.txt`. Change the dataroot and csv_file path to your own customized paths, then use the following line for training:

```
sh bash_run_train_mavoc.sh --opt configs/004_final_sareo_light_semisuper_simpledual_focalloss_aug.yml
```

After the training process is finished, inference the test images using the following commands:

```
python test_mavoc.py --opt configs/004_final_sareo_light_semisuper_simpledual_focalloss_aug.yml
```

