# VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset

- This is the official repository of VALOR which provides training & testing code and pretraining checkpoints.
- VALOR-32K dataset (annotation) can be downloaded from  [project page](https://casia-iva-group.github.io/projects/VALOR/download.html). Raw videos can be downloaded from YouTube.
- VALOR-1M will be released after paper is accepted.

<div align=center><img src=img/img_model.png/></div>

## Building Environment
- VALOR is implemented based on Pytorch. 

> We use pytorch-1.9.0 and cuda-11.1. 

Other version could be also compatible.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
---
- build apex. 
```
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
---
- install needed packages.
```
sh preinstall.sh
```

## Download Checkpoints
> Put pretrained_weights dir under main path. (VALOR/pretrained_weights)

- [pretrained_weights](https://drive.google.com/file/d/1KyqOzQIzNcL1Q9uEGmDECHfU-8CCd4kk/view?usp=sharing) 
(BERT,CLIP,VideoSwin).

---

- ### VALOR models.

    Put VALOR-base and VALOR-large under the output dir.<br> 
        ```VALOR/output/VALOR-base``` <br>
        ```VALOR/output/VALOR-large```

| Model   | Pretrained Ckpt | Finetuned Ckpt on MSRVTT-Retrieval | Finetuned Ckpt on MSRVTT-Caption |
|---------|-----------------|------------------------------------|----------------------------------|
| VALOR-B |       [VALOR-base](https://drive.google.com/file/d/1l-G255vTPt6XKMK-Ln42Jz_raGzipL84/view?usp=sharing)      |                [VALOR_base_msr_ret.pt](https://drive.google.com/file/d/1-YrVWKJUwKHTocikN4bvo62Wu78aZhHb/view?usp=sharing)                |               [VALOR_base_msr_cap.pt](https://drive.google.com/file/d/1-mzhin9n9iSCDMjMpAXS8jT2vUHlFN5f/view?usp=sharing)               |
| VALOR-L |       [VALOR-large](https://drive.google.com/file/d/1qFb9ejO-FLUTfZQkW_IFJrFEWyxjs72k/view?usp=sharing)      |                [VALOR_large_msr_ret.pt](https://drive.google.com/file/d/1-XViAyPRovm5WaaN5f1Heh9gPXdqcKBY/view?usp=sharing)                |               [VALOR_large_msr_cap.pt](https://drive.google.com/file/d/1-i_1yfZUMIXbTL8PSM0WtmSguu2eN-kk/view?usp=sharing)               |



## Prepare Datasets
VALOR is pretrained and tested on multiple vision-language, audio-language and audiovisual-language datasets. 
e.g. 
- PRETRAIN: VALOR-1M
- WebVid-2.5M
- CC-3M (VALOR-base)

TEST: 
- VALOR-32K
-  MSRVTT
- MSVD
- DiDeMo
- LSMDC
- ActivityNet
- VATEX 
- AudioCaps
- ClothoV1
- TGIF-Frame
- MSCOCO
- VQAV2...

We here take ```MSRVTT as an example``` to show the data processing procedures, other datasets take a similar way.

- make dir   `VALOR/datasets/MSRVTT`

- download raw videos from website, and put them in `MSRVTT/raw_videos`

- extract video frames (.jpg) and audio files (.wav). 
    - Utilizing utils/extract_frame_and_wav_multiprocess.py 
        > (Note: VALOR use this offline extracted frames and audios for training and testing for it's fast I/O speed. <br> You may adjust to read raw videos via decord library, and need to change VideoMapper and AudioMapper classes in data/data.py.)


- prepare id_files
    - standardsplit_train_id.json, 
    - standardsplit_test_id.json, 
    - 1KAsplit_train_id.json, 
    - 1KAsplit_test_id.json). 
    
    The format is `List(Str) ['video0', 'video1', ...].` <br>
    The former two are for video captioning and video qa, while the latter two are for video retrieval.  
    
- prepare txt_mapper.json. txt_mapper files map videoIDs to its descriptions.

    > Format `{'video0':['desc1','desc2',...'desc20']}`.

    - For VideoQA task, 

         > the format is `{'video0':[{'question':'what color is ...?', 'answer':'red'},{'question':'Is the boy ...?', 'answer':'yes'}]}`
    - prepare caption_annotation.json. : This file is used for computing caption metrics. 

         > format: `[{'video_id':'video0','caption','A boy is ...'}, {'video_id':'video1','caption','A girl is ...'}]`   
    

The processed  dataset path should be as follows: <br>
 ```
    ├── datasets
    │   ├── msrvtt
    │   │   ├── raw_videos
    │   │   │    ├── video0.mp4
    │   │   │    └── video1.mp4
    │   │   ├── frames_fps4
    │   │   │    ├── video0
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   │    └── video1
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   ├── audio_22050hz
    │   │   │    ├── video1.wav
    │   │   │    └── video3.wav
    │   │   ├── standardsplit_train_id.json
    │   │   ├── standardsplit_test_id.json
    │   │   ├── 1KAsplit_train_id.json
    │   │   ├── 1KAsplit_test_id.json
    │   │   ├── txt_mapper.json
    │   │   ├── txt_mapper_1kAsplit_test.json    
    │   │   ├── txt_mapper_vqa.json    
    │   │   └── caption_annotation.json    
```


We provide processed json files for most finetuneing datasets [here](https://drive.google.com/file/d/1pWym3bMNW_WrOZCi5Ls-wKFpaPMbLOio/view?usp=sharing)

> and you only need to download and extract raw videos of each dataset.



## Finetune  Model

- finetune retrieval tasks
```
sh scripts/finetune_ret.sh $pretrain_path(output/VALOR_base)
```
---
What I'm interested in
- finetune captioning tasks
```
sh scripts/finetune_cap.sh $pretrain_path(output/VALOR_base)
```

---
- finetune QA tasks
```
sh scripts/finetune_qa.sh $pretrain_path(output/VALOR_base)
```
The finetuning output path will be the subdir of `$pretrain_path`

## Test Model
For example, the cmd for finetuning retrieval model in ```scripts/finetune_ret.sh``` is as follows:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
--pretrain_dir $basedir \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/ret-msrvtt-lr2e-5-bs64-epoch5'   \
--learning_rate 2e-5  \
--train_video_sample_num 4 \
--test_video_sample_num 8  \
--save_best true \
```

if you want to test model, just add following two rows to the cmd:
```
--zero_shot \
--checkpoint $checkpoint_save_path(.pt)
```

## Pretrain Model
```
sh scripts/pretrain.sh
```


## Customize
VALOR's framework is easy to expand new tasks/datasets. what you need to do is 

1. prepare dataset as illustrated above
2. write config file (copy a config file and change 'data_cfg')

- In development stage, you can simply use cmd to overwrite config file. The most important args are :
    1. --learning_rate
    2. --train_batch_size
    4. --train_video_sample_num
    5. --test_video_sample_num
    6. --train_audio_sample_num
    7. --test_audio_sample_num
    8. --video_resolution
    9. --train_epoch
    10. --train_task
    11. --test_task

 <br>

- To control task and used modality group, you can rewrite train_task by 
`'task%modality_group1%modality_group2'`  <br>

> For example: finetuning text-to-audio retrieval  `'ret%ta'` <br> finetuning text-to-video retrieval  `'ret%tv' or 'ret%tva'` 
             

- Other settings
1. --fp16 (default: `True`)
2. --checkpointing (default: `False`)




## Citation

If you find this code useful for your research, please consider citing:


```
@article{chen2023valor,
  title={VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset},
  author={Chen, Sihan and He, Xingjian and Guo, Longteng and Zhu, Xinxin and Wang, Weining and Tang, Jinhui and Liu, Jing},
  journal={arXiv preprint arXiv:2304.08345},
  year={2023}
}
```

## License

MIT ---
