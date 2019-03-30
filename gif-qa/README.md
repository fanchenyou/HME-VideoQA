### Experiment on TGIF-QA

### Dataset
In this experiment, we use extended TGIF-QA dataset contains 165K QA pairs for the animated GIFs from the [[TGIF dataset]](https://github.com/YunseokJANG/tgif-qa)
We strictly follow their split of dataset in order to make fair comparison, though we did rewrite their Tensorflow based code with PyTorch.
Please cite this paper[[10]](https://arxiv.org/abs/1704.04497) if you use this dataset, and consider comparing your methods with the following great work ST-VQA
Please refer to their website for the detailed statistics of TGIF-QA dataset.


1. First, download their extracted _**Resnet_pool5, C3D_fc6**_ features [[here]](https://github.com/YunseokJANG/tgif-qa/blob/master/code/README.md),
move features to `data/feats`. They are quite big (37G and 73G each!!). Take your time to download them, and thanks for 
original author to share these files.

2. Second, download TGIF-QA question sets [[here]](https://github.com/YunseokJANG/tgif-qa/tree/master/dataset), 
and override `data/dataset`.

3. Optionally, download and extract glove.42B.300d.txt [[here]](http://nlp.stanford.edu/data/glove.42B.300d.zip)
and move to `data/Vocabulary`. We have included our word embeddings initialized with glove300D in this folder, with suffix .pkl. If you
want to try different embedding initialization, remove .pkl files and execute main.py once to 
automatically generate new vocabulary files.

4. Optionally, you can download original TGIFs and store in `data/gifs`. This helps visualization.



### Pre-trained models
We provide our pre-trained models to replicate the reported numbers in our paper.
1. Download from [[here]](https://drive.google.com/drive/folders/1T37IWDiNY--9xZszikHxINgUQbxMyH4y?usp=sharing) and override current empty saved_models folder.

We found that different platforms and different PyTorch versions produce slightly different
accuracy numbers but the difference less than 1%.



### Train, validate, and test
For training and validating, to perform any of four TGIF-QA tasks, execute the following command
~~~~
python main.py --task=[Count|Action|Trans|FrameQA] 
~~~~

For testing, just add a --test=1 flag, such as
~~~~
python main.py --task=[Count|Action|Trans|FrameQA] --test=1
~~~~

Please modify train.py to test your own models. 
Current we use default models we provided in previous steps.

## Quantitative Results

| Model                                    | Repetition Count <br/> (L2 loss) | Repeating Action <br/> (Accuracy) | State Transition <br/> (Accuracy) | Frame QA <br/> (Accuracy) |
| ---------------------------------------- | ---------------------: | --------------------------: | --------------------------: | ------------------: |               
| ST-VQA[[10]](https://arxiv.org/abs/1704.04497)                                     |                 4.28 |                       0.608 |                       0.671 |               0.493 |
| CO-Mem[[5]](https://arxiv.org/abs/1803.10906)                         |                 4.10 |                       0.682 |                       0.743 |               0.515 |
| Ours                                         |                **4.02** |                       **0.739** |                       **0.778** |              **0.538** |


