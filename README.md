## EmotionFlow
------
Source code for ICASSP2022 paper: EmotionFlow: Capture the Dialogue Level Emotion Transitions

### Required Packages:
------
transformers=4.14.1

torch=1.8

vocab=0.0.5

numpy

tqdm

sklearn

pickle

pandas


### Quick start:
------
download MELD dataset from https://github.com/declare-lab/MELD/ and save to ./MELD

#### Training
------
```
python train.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 -bert_path roberta-[base, large] -epochs [20, 5]
```

#### Evaluation
------
```
python train.py -te -ft -bsz 1 -dpt 0.3 -bert_path roberta-[base, large]
```

#### Results
------

| model                     | weighted-F1 |
|---------------------------|-------------|
| EmotionFlow-roberta-base  | 65.05       |
| EmotionFlow-roberta-large | 66.50       |

