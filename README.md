# CM-NAS

Official Pytorch code of paper [CM-NAS: Cross-Modality Neural Architecture Search for Visible-Infrared Person Re-Identification](https://arxiv.org/abs/2101.08467) in ICCV2021.

Visible-Infrared person re-identification (VI-ReID) aims to match cross-modality pedestrian images, breaking through the limitation of single-modality person ReID in dark environment. In order to mitigate the impact of large modality discrepancy, existing works manually design various two-stream architectures to separately learn modalityspecific and modality-sharable representations. Such a manual design routine, however, highly depends on massive experiments and empirical practice, which is time consuming and labor intensive. In this paper, we systematically study the manually designed architectures, and identify that appropriately separating Batch Normalization (BN) layers is the key to bring a great boost towards crossmodality matching. Based on this observation, the essential objective is to find the optimal separation scheme for each BN layer. To this end, we propose a novel method, named Cross-Modality Neural Architecture Search (CM-NAS). It consists of a BN-oriented search space in which the standard optimization can be fulfilled subject to the cross-modality task. Equipped with the searched architecture, our method outperforms state-of-the-art counterparts in both two benchmarks, improving the Rank-1/mAP by **6.70%**/**6.13%** on SYSU-MM01 and by **12.17%**/**11.23%** on RegDB.

## Requirements
Our experiments are conducted under the following environments:
- Python 3.7 <br>
- Pytorch == 1.3.1 <br>
- torchvision == 0.4.2 <br>

## Model Zoo
The searched configurations and the trained models can be downloaded in this [link](https://drive.google.com/drive/folders/1eLOrUYVAPTLT9BuUsgMCutRqMmhNgY8I).

Dataset | Protocol | Rank-1 | mAP | Protocol | Rank-1 | mAP | Trained Model
:---- | :----: | :----: | :----: | :----: | :----:| :----: | :----:
SYSU-MM01 | All-Single | 61.99% | 60.02% | Indoor-Single | 67.01% | 72.95% | [Google Drive](https://drive.google.com/drive/folders/1NlMPe8pneKiSAOl3VezNq9VWZcbnQwEC)
RegDB | Vis-to-Inf | 84.54% | 80.32% | Inf-to-Vis | 82.57% | 78.31% | [Google Drive](https://drive.google.com/drive/folders/1_afR3rtHlS-i7-M9BWOESRBxKp_w3r33)

Noet, the results may have some fluctuations caused by random spliting the datasets.

## Search
Codes will be released soon.

## Train
Before training, please download the searched configurations.

- For SYSU-MM01, first run [./utils/pre_process_sysu.py](https://github.com/JDAI-CV/CM-NAS/blob/main/utils/pre_process_sysu.py) to prepare data, then configure `data_root` path in [train_sysu.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/train_sysu.sh) and run [train_sysu.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/train_sysu.sh).
- For RegDB, configure `data_root` path in [train_regdb.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/train_regdb.sh) and run [train_regdb.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/train_regdb.sh).

## Test
Before testing, please download the searched configurations and the trained models.

- For SYSU-MM01, configure `data_root` path in [test_sysu.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/test_sysu.sh) and run [test_sysu.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/test_sysu.sh).
- For RegDB, configure `data_root` path in [test_regdb.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/test_regdb.sh) and run [test_regdb.sh](https://github.com/JDAI-CV/CM-NAS/blob/main/test_regdb.sh).

## License
CM-NAS is released under the Apache License 2.0. Please see the [LICENSE](https://github.com/JDAI-CV/CM-NAS/blob/main/LICENSE) file for more information.

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{Fu2021CMNAS,
  title     =  {CM-NAS: Cross-Modality Neural Architecture Search for Visible-Infrared Person Re-Identification},
  author    =  {Chaoyou Fu, Yibo Hu, Xiang Wu, Hailin Shi, Tao Mei and Ran He},
  booktitle =  {ICCV},
  year      =  {2021}
}
```

## Acknowledgements
This repo is based on the following repo, thank the authors a lot.
- [mangye16/Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)