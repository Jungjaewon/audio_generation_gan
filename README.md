# Audio_Generation_gan

This repository is a practice result of sound generation using SC09 dataset.

## Requirements
* python3.6+
* pytorch 1.6.0
* others.

## Usage
training a model
```bash
python3 main.py --config config.yml
```

testing a model
```bash
Not implmented yet
```

## Comments
WGAN_GP loss is used to generate audio and data space is too big. The model can not generate fake audios well.

## Reference
1. http://blog.naver.com/PostView.nhn?blogId=honeycomb-tech&logNo=221632925662&categoryNo=11&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
