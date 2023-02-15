import pandas as pd
import os
import os.path as osp
from glob import glob
from tqdm import tqdm

dataroot = '/path/to/dataset/root'

SELECT_SPLIT = ['TRAIN', 'VALID'] # 'TRAIN', 'VALID' or 'TEST'

if 'TRAIN' in SELECT_SPLIT:
    # prepare train csv file
    train_dir = 'train_images'
    train_df = pd.DataFrame(columns=["image_id", "sar_path", "eo_path", "class_id"])
    for clsid in range(10):
        print(f"[processing] class {clsid} in train set")
        img_paths = glob(osp.join(dataroot, train_dir, str(clsid), "*.png"))
        imids = set([osp.basename(img_path).split('.')[0].split("_")[1] for img_path in img_paths])
        pbar = tqdm(enumerate(imids), total=len(imids))
        for idx, imid in pbar:
            sar_path = osp.join(dataroot, train_dir, str(clsid), f"SAR_{imid}.png")
            eo_path = osp.join(dataroot, train_dir, str(clsid), f"EO_{imid}.png")
            sample_dict = {
                "image_id": imid,
                "sar_path": osp.join(train_dir, str(clsid), f"SAR_{imid}.png"),
                "eo_path": osp.join(train_dir, str(clsid), f"EO_{imid}.png"),
                "class_id": clsid
            }
            train_df = train_df.append(sample_dict, ignore_index=True)
            pbar.set_postfix(Idx=str(idx).zfill(8), Imid=str(imid).zfill(8))

    train_df.to_csv('train_dataset_info.csv')


if 'VALID' in SELECT_SPLIT:
    valid_dir = 'valid_images'
    valid_df = pd.DataFrame(columns=["image_id", "sar_path", "eo_path"])
    # prepare validation csv file
    img_paths = glob(osp.join(dataroot, valid_dir, 'valid_eo', "*.png"))
    imids = set([osp.basename(img_path).split('.')[0].split("_")[1] for img_path in img_paths])
    pbar = tqdm(enumerate(imids), total=len(imids))
    for idx, imid in pbar:
        sar_path = osp.join(dataroot, valid_dir, 'valid_sar', f"SAR_{imid}.png")
        eo_path = osp.join(dataroot, valid_dir, 'valid_eo', f"EO_{imid}.png")
        sample_dict = {
            "image_id": imid,
            "sar_path": osp.join(valid_dir, 'valid_sar', f"SAR_{imid}.png"),
            "eo_path": osp.join(valid_dir, 'valid_eo', f"EO_{imid}.png")
        }
        valid_df = valid_df.append(sample_dict, ignore_index=True)
        pbar.set_postfix(Idx=str(idx).zfill(8), Imid=str(imid).zfill(8))

    valid_df.to_csv('valid_dataset_info.csv')


if 'TEST' in SELECT_SPLIT:
    # prepare test csv file
    test_dir = 'test_images'
    test_df = pd.DataFrame(columns=["image_id", "sar_path", "eo_path"])
    img_paths = glob(osp.join(dataroot, test_dir, 'test_eo', "*.png"))
    imids = set([osp.basename(img_path).split('.')[0].split("_")[1] for img_path in img_paths])
    pbar = tqdm(enumerate(imids), total=len(imids))
    for idx, imid in pbar:
        sar_path = osp.join(dataroot, test_dir, 'test_sar', f"SAR_{imid}.png")
        eo_path = osp.join(dataroot, test_dir, 'test_eo', f"EO_{imid}.png")
        sample_dict = {
            "image_id": imid,
            "sar_path": osp.join(test_dir, 'test_sar', f"SAR_{imid}.png"),
            "eo_path": osp.join(test_dir, 'test_eo', f"EO_{imid}.png")
        }
        test_df = test_df.append(sample_dict, ignore_index=True)
        pbar.set_postfix(Idx=str(idx).zfill(8), Imid=str(imid).zfill(8))

    test_df.to_csv('test_dataset_info.csv')

