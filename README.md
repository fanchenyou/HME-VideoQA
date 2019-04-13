### Heterogeneous Memory Enhanced Multimodal Attention Model for VideoQA 
### (HME-VideoQA)

This is the PyTorch Implementation of 
* Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, Heng Huang. *Heterogeneous Memory Enhanced Multimodal Attention Model for VideoQA*. In *CVPR*, 2019. 
[[link]](https://arxiv.org/pdf/1904.04357.pdf)

 ```
@inproceedings{fan-CVPR-2019,
    author    = {Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, Heng Huang},
    title     = "{Heterogeneous Memory Enhanced Multimodal Attention Model for Video Question Answering}"
    booktitle = {CVPR},
    year      = 2019
}
```

Multiple-choice Task  
![Task](/pics/multi-choice.png)  

Open-ended Task
![Task](/pics/open-ended.png)


### Architecture
![Network](/pics/mmnet.png)

### Datasets
TGIF-QA [[here]](https://github.com/YunseokJANG/tgif-qa)

MSVD-QA [[here]](https://github.com/xudejing/VideoQA)


### Requirements
Python = 2.7
 
PyTorch = 1.0 [[here]](https://pytorch.org/)

GPU training with 4G+ memory, testing with 1G+ memory.


