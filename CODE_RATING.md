# 🍷 Code Sommelier 品鉴报告
> **生成时间**: 2026-02-02 15:29:08

## 1. 庄园综合评级 (Overall Assessment)
- **综合评分**: `80.92 / 100`
- **品质等级**: **B (Village / 村庄级)**
- **品鉴结论**: *整体骨架健康，但部分细节略显粗糙，建议适度醒酒（重构）以释放潜力。*
- **样本数量**: 75 个文件

## 2. 葡萄园地图 (Vineyard Map)
```text
├── CHANGELOG.md
├── CONTRIBUTING.md
├── DA2
│   ├── DA-2K.md
│   ├── LICENSE
│   ├── README.md
│   ├── app.py
│   ├── assets
│   │   ├── DA-2K.png
│   │   ├── examples
│   │   │   ├── demo01.jpg
│   │   │   ├── demo02.jpg
│   │   │   ├── demo03.jpg
│   │   │   ├── demo04.jpg
│   │   │   ├── demo05.jpg
│   │   │   ├── demo06.jpg
│   │   │   ├── demo07.jpg
│   │   │   ├── demo08.jpg
│   │   │   ├── demo09.jpg
│   │   │   ├── demo10.jpg
│   │   │   ├── demo11.jpg
│   │   │   ├── demo12.jpg
│   │   │   ├── demo13.jpg
│   │   │   ├── demo14.jpg
│   │   │   ├── demo15.jpg
│   │   │   ├── demo16.jpg
│   │   │   ├── demo17.jpg
│   │   │   ├── demo18.jpg
│   │   │   ├── demo19.jpg
│   │   │   └── demo20.jpg
│   │   ├── examples_video
│   │   │   ├── basketball.mp4
│   │   │   └── ferris_wheel.mp4
│   │   └── teaser.png
│   ├── checkpoints
│   │   └── depth_anything_v2_vits.pth
│   ├── depth_anything_v2
│   │   ├── dinov2.py
│   │   ├── dinov2_layers
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── dpt.py
│   │   └── util
│   │       ├── blocks.py
│   │       └── transform.py
│   ├── metric_depth
│   │   ├── README.md
│   │   ├── assets
│   │   │   └── compare_zoedepth.png
│   │   ├── dataset
│   │   │   ├── hypersim.py
│   │   │   ├── kitti.py
│   │   │   ├── splits
│   │   │   │   ├── hypersim
│   │   │   │   │   ├── train.txt
│   │   │   │   │   └── val.txt
│   │   │   │   ├── kitti
│   │   │   │   │   └── val.txt
│   │   │   │   └── vkitti2
│   │   │   │       └── train.txt
│   │   │   ├── transform.py
│   │   │   └── vkitti2.py
│   │   ├── depth_anything_v2
│   │   │   ├── dinov2.py
│   │   │   ├── dinov2_layers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── attention.py
│   │   │   │   ├── block.py
│   │   │   │   ├── drop_path.py
│   │   │   │   ├── layer_scale.py
│   │   │   │   ├── mlp.py
│   │   │   │   ├── patch_embed.py
│   │   │   │   └── swiglu_ffn.py
│   │   │   ├── dpt.py
│   │   │   └── util
│   │   │       ├── blocks.py
│   │   │       └── transform.py
│   │   ├── depth_to_pointcloud.py
│   │   ├── dist_train.sh
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   ├── train.py
│   │   └── util
│   │       ├── dist_helper.py
│   │       ├── loss.py
│   │       ├── metric.py
│   │       └── utils.py
│   ├── requirements.txt
│   ├── run.py
│   ├── run_video.py
│   └── use_model.py
├── LICENSE.txt
├── README.md
├── config
│   ├── dataset
│   │   ├── data_diode_all.yaml
│   │   ├── data_eth3d.yaml
│   │   ├── data_hypersim_train.yaml
│   │   ├── data_hypersim_val.yaml
│   │   ├── data_kitti_eigen_test.yaml
│   │   ├── data_kitti_val.yaml
│   │   ├── data_nyu_test.yaml
│   │   ├── data_nyu_train.yaml
│   │   ├── data_scannet_val.yaml
│   │   ├── data_vkitti_train.yaml
│   │   ├── data_vkitti_val.yaml
│   │   ├── dataset_train.yaml
│   │   ├── dataset_val.yaml
│   │   └── dataset_vis.yaml
│   ├── logging.yaml
│   ├── model_sdv2.yaml
│   ├── train_debug.yaml
│   ├── train_marigold.yaml
│   └── wandb.yaml
├── data_split
│   ├── diode
│   │   ├── diode_val_all_filename_list.txt
│   │   ├── diode_val_indoor_filename_list.txt
│   │   └── diode_val_outdoor_filename_list.txt
│   ├── eth3d
│   │   └── eth3d_filename_list.txt
│   ├── hypersim
│   │   ├── filename_list_test_filtered.txt
│   │   ├── filename_list_train_filtered.txt
│   │   ├── filename_list_val_filtered.txt
│   │   ├── filename_list_val_filtered_small_80.txt
│   │   └── selected_vis_sample.txt
│   ├── kitti
│   │   ├── eigen_test_files_with_gt.txt
│   │   ├── eigen_val_from_train_800.txt
│   │   └── eigen_val_from_train_sub_100.txt
│   ├── nyu
│   │   └── labeled
│   │       ├── filename_list_test.txt
│   │       ├── filename_list_train.txt
│   │       └── filename_list_train_small_100.txt
│   ├── scannet
│   │   └── scannet_val_sampled_list_800_1.txt
│   └── vkitti
│       ├── vkitti_train.txt
│       └── vkitti_val.txt
├── doc
│   ├── apdepth-v1-1
│   │   ├── infer.png
│   │   └── train.png
│   ├── apdepth-v1-2
│   │   ├── infer.png
│   │   └── train.png
│   ├── badges
│   │   ├── badge-colab.svg
│   │   ├── badge-docker.svg
│   │   └── badge-website.svg
│   ├── changelog
│   │   └── all-framework.png
│   ├── cover.jpg
│   ├── cover.png
│   ├── main.jpg
│   └── teaser_collage_transparant.png
├── environment.yaml
├── eval.py
├── infer.py
├── input
│   ├── in-the-wild_example
│   │   ├── example_0.jpg
│   │   ├── example_1.jpg
│   │   ├── example_2.jpg
│   │   ├── example_3.jpg
│   │   ├── example_4.jpg
│   │   ├── example_5.jpg
│   │   ├── example_6.jpg
│   │   ├── example_7.jpg
│   │   └── images_inference.txt
│   ├── in-the-wild_example-1
│   │   ├── 00.jpg
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   ├── 03.jpg
│   │   ├── 04.jpg
│   │   ├── 05.jpg
│   │   ├── 06.jpg
│   │   ├── 07.jpg
│   │   └── 08.jpg
│   ├── in-the-wild_example-2
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   ├── 03.jpg
│   │   ├── 04.jpg
│   │   ├── 05.jpg
│   │   ├── 06.jpg
│   │   ├── 07.jpg
│   │   ├── 08.jpg
│   │   └── 09.jpg
│   └── in-the-wild_example-3
│       ├── highres-01.jpeg
│       ├── highres-02.jpeg
│       ├── highres-03.jpeg
│       ├── highres-04.jpeg
│       ├── inside-01.jpg
│       ├── inside-02.jpg
│       ├── inside-03.jpg
│       ├── outside-01.jpg
│       ├── outside-02.jpg
│       ├── outside-03.jpg
│       ├── synthetic-inside-01.jpg
│       ├── synthetic-inside-02.jpg
│       ├── synthetic-inside-03.jpg
│       ├── synthetic-inside-04.jpg
│       ├── synthetic-inside-05.jpg
│       ├── synthetic-outside-01.jpg
│       ├── synthetic-outside-02.jpg
│       └── synthetic-outside-03.jpg
├── marigold
│   ├── __init__.py
│   ├── marigold_pipeline.py
│   └── util
│       ├── batchsize.py
│       ├── ensemble.py
│       └── image_util.py
├── requirements++.txt
├── requirements+.txt
├── requirements.txt
├── run.py
├── script
│   ├── dataset_preprocess
│   │   └── hypersim
│   │       ├── README.md
│   │       ├── hypersim_util.py
│   │       ├── metadata_images_split_scene_v1.csv
│   │       └── preprocess_hypersim.py
│   ├── download_weights.sh
│   └── eval
│       ├── 00_test_all.sh
│       ├── 11_infer_nyu.sh
│       ├── 12_eval_nyu.sh
│       ├── 21_infer_kitti.sh
│       ├── 22_eval_kitti.sh
│       ├── 31_infer_eth3d.sh
│       ├── 32_eval_eth3d.sh
│       ├── 41_infer_scannet.sh
│       ├── 42_eval_scannet.sh
│       ├── 51_infer_diode.sh
│       ├── 52_eval_diode.sh
│       ├── 61_infer_hypersim.sh
│       └── 62_eval_hypersim.sh
├── src
│   ├── __init__.py
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── base_depth_dataset.py
│   │   ├── diode_dataset.py
│   │   ├── eth3d_dataset.py
│   │   ├── hypersim_dataset.py
│   │   ├── kitti_dataset.py
│   │   ├── mixed_sampler.py
│   │   ├── nyu_dataset.py
│   │   ├── scannet_dataset.py
│   │   └── vkitti_dataset.py
│   ├── trainer
│   │   ├── __init__.py
│   │   └── marigold_trainer.py
│   └── util
│       ├── alignment.py
│       ├── build_mlp.py
│       ├── config_util.py
│       ├── data_loader.py
│       ├── depth_transform.py
│       ├── logging_util.py
│       ├── loss.py
│       ├── lr_scheduler.py
│       ├── metric.py
│       ├── multi_res_noise.py
│       ├── seeding.py
│       └── slurm_util.py
└── train.py
```

## 3. 详细风味分析 (Detailed Notes)
| 文件名 | 语言 | 得分 | 等级 | 状态 |
| :--- | :---: | :---: | :---: | :---: |
| `dinov2.py` | Python | 31.0 | **D** | 🛑 |
| `dinov2.py` | Python | 31.0 | **D** | 🛑 |
| `loss.py` | Python | 42.0 | **D** | 🛑 |
| `block.py` | Python | 43.0 | **D** | 🛑 |
| `block.py` | Python | 43.0 | **D** | 🛑 |
| `metric.py` | Python | 45.0 | **D** | 🛑 |
| `base_depth_dataset.py` | Python | 53.0 | **D** | 🛑 |
| `transform.py` | Python | 56.0 | **D** | 🛑 |
| `ensemble.py` | Python | 57.0 | **D** | 🛑 |
| `marigold_trainer.py` | Python | 57.0 | **D** | 🛑 |
| `dpt.py` | Python | 58.0 | **D** | 🛑 |
| `dpt.py` | Python | 59.0 | **D** | 🛑 |
| `transform.py` | Python | 65.0 | **C** | ⚠️ |
| `transform.py` | Python | 65.0 | **C** | ⚠️ |
| `logging_util.py` | Python | 66.0 | **C** | ⚠️ |
| `depth_transform.py` | Python | 69.0 | **C** | ⚠️ |
| `marigold_pipeline.py` | Python | 71.0 | **C** | ⚠️ |
| `data_loader.py` | Python | 72.0 | **C** | ⚠️ |
| `blocks.py` | Python | 73.0 | **C** | ⚠️ |
| `blocks.py` | Python | 73.0 | **C** | ⚠️ |
| `swiglu_ffn.py` | Python | 77.0 | **B** | 🆗 |
| `swiglu_ffn.py` | Python | 77.0 | **B** | 🆗 |
| `train.py` | Python | 77.0 | **B** | 🆗 |
| `attention.py` | Python | 79.0 | **B** | 🆗 |
| `attention.py` | Python | 79.0 | **B** | 🆗 |
| `hypersim.py` | Python | 81.0 | **B** | 🆗 |
| `kitti_dataset.py` | Python | 82.0 | **B** | 🆗 |
| `app.py` | Python | 84.0 | **B** | 🆗 |
| `kitti.py` | Python | 84.0 | **B** | 🆗 |
| `vkitti2.py` | Python | 84.0 | **B** | 🆗 |
| `loss.py` | Python | 84.0 | **B** | 🆗 |
| `config_util.py` | Python | 84.0 | **B** | 🆗 |
| `lr_scheduler.py` | Python | 84.0 | **B** | 🆗 |
| `diode_dataset.py` | Python | 85.0 | **A** | 🆗 |
| `vkitti_dataset.py` | Python | 85.0 | **A** | 🆗 |
| `patch_embed.py` | Python | 86.0 | **A** | 🆗 |
| `patch_embed.py` | Python | 86.0 | **A** | 🆗 |
| `mixed_sampler.py` | Python | 86.0 | **A** | 🆗 |
| `dist_helper.py` | Python | 87.0 | **A** | 🆗 |
| `metric.py` | Python | 87.0 | **A** | 🆗 |
| `utils.py` | Python | 87.0 | **A** | 🆗 |
| `build_mlp.py` | Python | 87.0 | **A** | 🆗 |
| `image_util.py` | Python | 88.0 | **A** | 🆗 |
| `run.py` | Python | 90.0 | **A** | ✅ |
| `run.py` | Python | 90.0 | **A** | ✅ |
| `run_video.py` | Python | 90.0 | **A** | ✅ |
| `preprocess_hypersim.py` | Python | 90.0 | **A** | ✅ |
| `__init__.py` | Python | 90.0 | **A** | ✅ |
| `drop_path.py` | Python | 91.0 | **A** | ✅ |
| `drop_path.py` | Python | 91.0 | **A** | ✅ |
| `nyu_dataset.py` | Python | 91.0 | **A** | ✅ |
| `alignment.py` | Python | 91.0 | **A** | ✅ |
| `mlp.py` | Python | 92.0 | **A** | ✅ |
| `mlp.py` | Python | 92.0 | **A** | ✅ |
| `multi_res_noise.py` | Python | 92.0 | **A** | ✅ |
| `layer_scale.py` | Python | 94.0 | **A** | ✅ |
| `layer_scale.py` | Python | 94.0 | **A** | ✅ |
| `hypersim_util.py` | Python | 94.0 | **A** | ✅ |
| `eth3d_dataset.py` | Python | 94.0 | **A** | ✅ |
| `hypersim_dataset.py` | Python | 94.0 | **A** | ✅ |
| `scannet_dataset.py` | Python | 94.0 | **A** | ✅ |
| `seeding.py` | Python | 94.0 | **A** | ✅ |
| `slurm_util.py` | Python | 94.0 | **A** | ✅ |
| `depth_to_pointcloud.py` | Python | 95.0 | **S** | ✅ |
| `run.py` | Python | 95.0 | **S** | ✅ |
| `infer.py` | Python | 97.0 | **S** | ✅ |
| `batchsize.py` | Python | 97.0 | **S** | ✅ |
| `__init__.py` | Python | 97.0 | **S** | ✅ |
| `__init__.py` | Python | 97.0 | **S** | ✅ |
| `__init__.py` | Python | 100.0 | **S** | ✅ |
| `__init__.py` | Python | 100.0 | **S** | ✅ |
| `use_model.py` | Python | 100.0 | **S** | ✅ |
| `eval.py` | Python | 100.0 | **S** | ✅ |
| `__init__.py` | Python | 100.0 | **S** | ✅ |
| `train.py` | Python | 100.0 | **S** | ✅ |

## 4. 酿造师建议 (Winemaker's Suggestions)
### 📄 `dinov2.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 4.8% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'named_apply' 深度 25 层
- 🏗️ 🏗️ 嵌套过深: 函数 'init_weights_vit_timm' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_small' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_base' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_large' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_giant2' 深度 13 层
- 🏷️ 🎨 色泽偏差: 函数 'DINOv2' 建议使用 snake_case
- 🏗️ 🏗️ 嵌套过深: 函数 'DINOv2' 深度 13 层
- 📏 📏 酒体过重: 函数 '__init__' 长达 125 行 (建议拆分)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (22个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 26 层
- 🏗️ 🏗️ 嵌套过深: 函数 'init_weights' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'interpolate_pos_encoding' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 'prepare_tokens_with_masks' 深度 21 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward_features_list' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward_features' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_intermediate_layers_not_chunked' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_intermediate_layers_chunked' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_intermediate_layers' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 11 层

### 📄 `dinov2.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 4.8% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'named_apply' 深度 25 层
- 🏗️ 🏗️ 嵌套过深: 函数 'init_weights_vit_timm' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_small' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_base' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_large' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'vit_giant2' 深度 13 层
- 🏷️ 🎨 色泽偏差: 函数 'DINOv2' 建议使用 snake_case
- 🏗️ 🏗️ 嵌套过深: 函数 'DINOv2' 深度 13 层
- 📏 📏 酒体过重: 函数 '__init__' 长达 125 行 (建议拆分)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (22个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 26 层
- 🏗️ 🏗️ 嵌套过深: 函数 'init_weights' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'interpolate_pos_encoding' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 'prepare_tokens_with_masks' 深度 21 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward_features_list' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward_features' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_intermediate_layers_not_chunked' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_intermediate_layers_chunked' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_intermediate_layers' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 11 层

### 📄 `loss.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 9.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'get_loss' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_smooth_loss' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'create_high_pass_weights' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 27 层

### 📄 `block.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 7.1% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'drop_add_residual_stochastic_depth' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_branges_scales' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'add_residual' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_attn_bias_and_cat' 深度 27 层
- 🏗️ 🏗️ 嵌套过深: 函数 'drop_add_residual_stochastic_depth_list' 深度 26 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (15个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward_nested' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 'attn_residual_func' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ffn_residual_func' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'attn_residual_func' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ffn_residual_func' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'attn_residual_func' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ffn_residual_func' 深度 13 层

### 📄 `block.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 7.1% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'drop_add_residual_stochastic_depth' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_branges_scales' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'add_residual' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_attn_bias_and_cat' 深度 27 层
- 🏗️ 🏗️ 嵌套过深: 函数 'drop_add_residual_stochastic_depth_list' 深度 26 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (15个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward_nested' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 'attn_residual_func' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ffn_residual_func' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'attn_residual_func' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ffn_residual_func' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'attn_residual_func' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ffn_residual_func' 深度 13 层

### 📄 `metric.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 3.8% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'abs_relative_difference' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'squared_relative_difference' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'rmse_linear' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'rmse_log' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'log10' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'threshold_percentage' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'delta1_acc' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'delta2_acc' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'delta3_acc' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'i_rmse' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'silog_rmse' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'reset' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 'update' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'avg' 深度 7 层

### 📄 `base_depth_dataset.py` (等级: D)
- 🏗️ 🏗️ 嵌套过深: 函数 'read_image_from_tar' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_pred_name' 深度 16 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (14个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__getitem__' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_data_item' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_load_rgb_data' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_load_depth_data' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_data_path' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_image' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_rgb_file' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_valid_mask' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_training_preprocess' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_augment_data' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__del__' 深度 14 层

### 📄 `transform.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 6.1% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'apply_min_size' 深度 15 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (8个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'constrain_to_multiple_of' 深度 18 层
- 🔄 🕸️ 结构纠结: 函数 'get_size' 复杂度 11
- 🏗️ 🏗️ 嵌套过深: 函数 'get_size' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 40 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 19 层

### 📄 `ensemble.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 9.5% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'inter_distances' 深度 15 层
- 📏 📏 酒体过重: 函数 'ensemble_depth' 长达 157 行 (建议拆分)
- 🏗️ ⚖️ 成分复杂: 函数 'ensemble_depth' 参数过多 (9个)
- 🔄 🕸️ 结构极其纠结: 函数 'ensemble_depth' 复杂度 26
- 🏗️ 🏗️ 嵌套过深: 函数 'ensemble_depth' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'init_param' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'align' 深度 18 层
- 🏗️ 🏗️ 嵌套过深: 函数 'ensemble' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'cost_fn' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'compute_param' 深度 15 层

### 📄 `marigold_trainer.py` (等级: D)
- 🏗️ 🏗️ 嵌套过深: 函数 '_replace_unet_conv_in' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'train' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'encode_depth' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'stack_depth_images' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_train_step_callback' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'validate' 深度 23 层
- 🏗️ 🏗️ 嵌套过深: 函数 'visualize' 深度 15 层
- 📏 📏 酒体略重: 函数 'validate_single_dataset' 长度 88 行
- 🏗️ 🏗️ 嵌套过深: 函数 'validate_single_dataset' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_next_seed' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'save_checkpoint' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'load_checkpoint' 深度 24 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_backup_ckpt_name' 深度 6 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (12个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 11 层

### 📄 `dpt.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 0.8% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '_make_fusion_block' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 6 层
- 📏 📏 酒体略重: 函数 '__init__' 长度 76 行
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 28 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 25 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 28 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 30 层
- 🏗️ 🏗️ 嵌套过深: 函数 'infer_image' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 'infer_batch' 深度 24 层
- 🏗️ 🏗️ 嵌套过深: 函数 'image2tensor' 深度 25 层

### 📄 `dpt.py` (等级: D)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '_make_fusion_block' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 6 层
- 📏 📏 酒体略重: 函数 '__init__' 长度 75 行
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 28 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 25 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 28 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 30 层
- 🏗️ 🏗️ 嵌套过深: 函数 'infer_image' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 'image2tensor' 深度 25 层

### 📄 `transform.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 7.0% (建议 > 10%)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (8个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'constrain_to_multiple_of' 深度 18 层
- 🔄 🕸️ 结构纠结: 函数 'get_size' 复杂度 11
- 🏗️ 🏗️ 嵌套过深: 函数 'get_size' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 26 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 14 层

### 📄 `transform.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 7.0% (建议 > 10%)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (8个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'constrain_to_multiple_of' 深度 18 层
- 🔄 🕸️ 结构纠结: 函数 'get_size' 复杂度 11
- 🏗️ 🏗️ 嵌套过深: 函数 'get_size' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 26 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 14 层

### 📄 `logging_util.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 4.9% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'config_logging' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'init_wandb' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'log_slurm_job_id' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'load_wandb_job_id' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'save_wandb_job_id' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'eval_dic_to_text' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'set_dir' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 'log_dic' 深度 17 层

### 📄 `depth_transform.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 6.8% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'get_depth_normalizer' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'denormalize' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'scale_back' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'denormalize' 深度 9 层

### 📄 `marigold_pipeline.py` (等级: C)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (9个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 34 层
- 📏 📏 酒体过重: 函数 '__call__' 长达 164 行 (建议拆分)
- ⚠️ ⚖️ 成分复杂: 函数 '__call__' 参数过多 (10个)
- 🔄 🕸️ 结构纠结: 函数 '__call__' 复杂度 15
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 18 层
- 🏗️ 🏗️ 嵌套过深: 函数 'encode_empty_text' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'single_infer' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 'encode_rgb' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'decode_depth' 深度 13 层

### 📄 `data_loader.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 5.4% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'skip_first_batches' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__iter__' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__len__' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__iter__' 深度 8 层

### 📄 `blocks.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '_make_scratch' 深度 29 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 25 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (8个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 27 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 22 层

### 📄 `blocks.py` (等级: C)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '_make_scratch' 深度 29 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 25 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (8个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 27 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 22 层

### 📄 `swiglu_ffn.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 7.9% (建议 > 10%)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 16 层

### 📄 `swiglu_ffn.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 7.9% (建议 > 10%)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 16 层

### 📄 `train.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 📏 📏 酒体过重: 函数 'main' 长达 165 行 (建议拆分)
- 🔄 🕸️ 结构极其纠结: 函数 'main' 复杂度 24
- 🏗️ 🏗️ 嵌套过深: 函数 'main' 深度 43 层

### 📄 `attention.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 9.6% (建议 > 10%)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 24 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 15 层

### 📄 `attention.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 9.6% (建议 > 10%)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 24 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 15 层

### 📄 `hypersim.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'hypersim_distance_to_depth' 深度 27 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__getitem__' 深度 15 层

### 📄 `kitti_dataset.py` (等级: B)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_load_rgb_data' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_load_depth_data' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'kitti_benchmark_crop' 深度 12 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_valid_mask' 深度 15 层

### 📄 `app.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'predict_depth' 深度 7 层
- 🏗️ 🏗️ 嵌套过深: 函数 'on_submit' 深度 17 层

### 📄 `kitti.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 1.8% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__getitem__' 深度 16 层

### 📄 `vkitti2.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 1.9% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__getitem__' 深度 20 层

### 📄 `loss.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 15 层

### 📄 `config_util.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 8.2% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'recursive_load_config' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 'find_value_in_omegaconf' 深度 17 层

### 📄 `lr_scheduler.py` (等级: B)
- 📝 🍷 余味不足: 注释率仅为 4.2% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__call__' 深度 14 层

### 📄 `diode_dataset.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_npy_file' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_data_path' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_data_item' 深度 19 层

### 📄 `vkitti_dataset.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_load_rgb_data' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_load_depth_data' 深度 16 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_valid_mask' 深度 15 层

### 📄 `patch_embed.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'make_2tuple' 深度 6 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 'flops' 深度 23 层

### 📄 `patch_embed.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'make_2tuple' 深度 6 层
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 22 层
- 🏗️ 🏗️ 嵌套过深: 函数 'flops' 深度 23 层

### 📄 `mixed_sampler.py` (等级: A)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__iter__' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__getitem__' 深度 6 层

### 📄 `dist_helper.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 2.4% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'setup_distributed' 深度 15 层

### 📄 `metric.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'eval_depth' 深度 25 层

### 📄 `utils.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'init_log' 深度 12 层

### 📄 `build_mlp.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)
- 🏗️ 🏗️ 嵌套过深: 函数 'build_mlp_' 深度 14 层

### 📄 `image_util.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'colorize_depth_maps' 深度 17 层
- 🏗️ 🏗️ 嵌套过深: 函数 'chw2hwc' 深度 10 层
- 🏗️ 🏗️ 嵌套过深: 函数 'resize_max_res' 深度 20 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_tv_resample_method' 深度 14 层

### 📄 `run.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)

### 📄 `run.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)

### 📄 `run_video.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)

### 📄 `preprocess_hypersim.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 9.9% (建议 > 10%)

### 📄 `__init__.py` (等级: A)
- 📝 🍷 余味不足: 注释率仅为 0.0% (建议 > 10%)

### 📄 `drop_path.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'drop_path' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 7 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 10 层

### 📄 `drop_path.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'drop_path' 深度 14 层
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 7 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 10 层

### 📄 `nyu_dataset.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 7 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_get_valid_mask' 深度 12 层

### 📄 `alignment.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'align_depth_least_square' 深度 18 层
- 🏗️ 🏗️ 嵌套过深: 函数 'depth2disparity' 深度 11 层
- 🏗️ 🏗️ 嵌套过深: 函数 'disparity2depth' 深度 10 层

### 📄 `mlp.py` (等级: A)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层

### 📄 `mlp.py` (等级: A)
- ⚠️ ⚖️ 成分复杂: 函数 '__init__' 参数过多 (7个)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 15 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 8 层

### 📄 `multi_res_noise.py` (等级: A)
- 🔄 🕸️ 结构极其纠结: 函数 'multi_res_noise_like' 复杂度 17
- 🏗️ 🏗️ 嵌套过深: 函数 'multi_res_noise_like' 深度 20 层

### 📄 `layer_scale.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 13 层

### 📄 `layer_scale.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 13 层
- 🏗️ 🏗️ 嵌套过深: 函数 'forward' 深度 13 层

### 📄 `hypersim_util.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'tone_map' 深度 19 层
- 🏗️ 🏗️ 嵌套过深: 函数 'dist_2_depth' 深度 18 层

### 📄 `eth3d_dataset.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 14 层

### 📄 `hypersim_dataset.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层

### 📄 `scannet_dataset.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 '__init__' 深度 6 层
- 🏗️ 🏗️ 嵌套过深: 函数 '_read_depth_file' 深度 8 层

### 📄 `seeding.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'seed_all' 深度 7 层
- 🏗️ 🏗️ 嵌套过深: 函数 'generate_seed_sequence' 深度 9 层

### 📄 `slurm_util.py` (等级: A)
- 🏗️ 🏗️ 嵌套过深: 函数 'is_on_slurm' 深度 9 层
- 🏗️ 🏗️ 嵌套过深: 函数 'get_local_scratch_dir' 深度 8 层

### 📄 `depth_to_pointcloud.py` (等级: S)
- 📏 📏 酒体略重: 函数 'main' 长度 76 行
- 🏗️ 🏗️ 嵌套过深: 函数 'main' 深度 29 层

### 📄 `run.py` (等级: S)
- 🛡️ 🙈 掩耳盗铃: 第 210 行捕获了异常却未处理

### 📄 `infer.py` (等级: S)
- 🏗️ 🏗️ 嵌套过深: 函数 'check_directory' 深度 8 层

### 📄 `batchsize.py` (等级: S)
- 🏗️ 🏗️ 嵌套过深: 函数 'find_batch_size' 深度 19 层

### 📄 `__init__.py` (等级: S)
- 🏗️ 🏗️ 嵌套过深: 函数 'get_dataset' 深度 15 层

### 📄 `__init__.py` (等级: S)
- 🏗️ 🏗️ 嵌套过深: 函数 'get_trainer_cls' 深度 8 层

---
**优化指南**: 建议重点关注复杂度过高的函数，通过拆分模块来降低耦合度，提升代码的可读性。