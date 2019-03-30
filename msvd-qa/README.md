### Experiment on MSVD-QA

### Dataset
In this experiment, we use [[MSVD-QA dataset]](https://github.com/xudejing/VideoQA).
Please cite this paper[[27]](https://www.comp.nus.edu.sg/~xiangnan/papers/mm17-videoQA.pdf) if you use this dataset, 
and consider comparing your methods with the following great work of AMU network design.
Please refer to their website for the detailed statistics of this dataset.
You don't need to download their features, we pack them and organize in following way.


1. First, download our packed Resnet and C3D features [[here]](https://drive.google.com/file/d/1i-8kie6yEXbrR-P4mUF4YimJcHYPa7Go/view?usp=sharing),
move video_feature_20.h5 to `data/msvd_qa`.

2. Second, download word embedding file [[here]](https://drive.google.com/file/d/1G7aFy3QS_PlhicFZ0MMnAHMxFC94p6Pl/view?usp=sharing), 
and move to `data/msvd_qa`.

3. Optionally you can download original MSVD-QA videos. This helps visualization.



### Pre-trained models
We provide our pre-trained models to replicate the reported numbers in our paper.
1. Download from [[here]](https://drive.google.com/file/d/196-z0cP29IMFqI9wJ-RPjtypXm70fhKQ/view?usp=sharing) and override current empty saved_models folder.


### Train, validate, and test
For training and validating, execute the following command
~~~~
python main.py
~~~~

For testing, just add a --test=1 flag, such as
~~~~
python main.py --test=1
~~~~

Please modify train.py to test your own models. 
Current we use default models we provided in previous steps.


