# Child Speech  Classifier

- Automatic child speech classification system based on a multilingual Large Language Model (LLMs)

### Goal

- We implement pre-trained transformer-based solutions to distinguish 
between L2 learner proclivities and underlying language impairment 
through downstream classification tasks.

## Instructions 

To use this classifier, run *main.py*

There are two initial settings - depending on the type of GPU you have, you can specify manually CUDA or METAL (msp) in the 
gpu_settings.py - The default is CUDA. 
The GPU settings can also bee toggled in the individual paradigm files, *binary_one.py*, *multiclass_two.py* 
and *augmented_three.py*. 

The program then gives you the following options (*see below for explanations*) :


#### Choice of model: 
                    - Binary for binary classification
                    - Multiclass for multiclass classification
                    - Augmented for models with hybrid data formation


#### Choice of language: 
                    - English Dataset
                    - Farsi Dataset
                    - Greek Dataset
                    - Multilingual Dataset


#### Choice of Model Depth: 
                    - Vanilla
                    - Shallow
                    - Inherent

Once it's done training you are prompted to choose a directory for your trained model. 

## Datasets
<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/Datasets.png" width="500" ></a>
*Please DM for curated datasets* 

### *English*

Data Distribution: 

<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/english_SBS.png" width="300" ></a>
<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/english_HOS.png" width="300" ></a>

### *Farsi*

Data Distribution:

<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/farsi_SBS.png" width="300" ></a>
<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/farsi_HOS.png" width="300" ></a>

### *Greek*

Data Distribution:

<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/greek_SBS.png" width="300" ></a>
<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/greek_HOS.png" width="300" ></a>

## <ins>Model Paradigms</ins>

### *Binary Classification*
The goal of this first experiment is to assess the mBERT models’ 
performance as a multilingual binary classifier. Given a target 
sentence a model should accurately discern the felicity of a given 
input by ascribing a score of either 0 or 1.

 * See https://github.com/Herrmandela/TOLD_mBERT_Classifier for Binary Classification Jupyter Notebook. 

### *Multiclass Classification*

The second experiment is largely built on the script used in the
first experiment with the main difference being that it deploys a
multi-classification feature in addition to the binary classification head.
Just like in the first experiment we’re trying to determine how 
multilingual mBERT is and whether it is suitable for our ultimate 
task of constructing a pipeline to evaluate children on the basis of
tasks designed to detect language impairment. Here, we’re not only 
interested in whether the child-speech utterance is correct or incorrect,
but also, to which degree it is correct and what type of sentence 
structure the target sentence has.


### *Augmentation and Hybrid Data Formation*

There is a prevailing disparity when it comes to language representation in NLP
applications. The lionshare of the attention has been allocated to languages like
English, that are over-represented in the literature as is. 
The literature also states that as more datapoints are provided a model 
the better it can preform in downstream tasks. 
The amount of data we use to fine-tune our models is miniscule compared to 
the vast amounts of multilingual data that was used to pretrain mBERT.
The training data we use is not multilingual, while the test data is composed 
of samples from our three datasets. 
Our specific use case requires models that are capable of cross-linguistic GED.

Thus, we want to determine how many data points, of which language, 
at what parameter settings yields the best crosslingual model for our tasks.
Building on the previous paradigms, in this final part we will evaluate the 
best performing models by comparing their performance with and without 
augmented data. 
We resort to the creation of synthetic data and seek to 
create the favorable conditions allotted by increased datapoints.

Synthetic data is formed using the NlpAug module which creates realistic data 
by mimicking behavior that leads to the formation of corrupted recitation patterns
in natural settings. NlpAug can be configured to incorporate word order violations,
inappropriate synonym insertion and word omission. 
After the production of the synthetic data, it is automatically incorporated 
to the training data forming cohesive hybrid dataset.

 * See https://github.com/Herrmandela/CELF_mBERT_Classifier for Binary Classification Jupyter Notebook.

<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/augFigure.png" width="500" ></a>

## Model Depth


### *Vanilla*

No layers frozen

### *Shallow*

Shallow Layers Frozen + Output Layer

### *Inherent*

All Layers Frozen Except for Input and Output Layers 

<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/layerFreeze.png" width="500" ></a>


## Metrics

### *Sample Sentence*

<a href="url"><img src="https://github.com/Herrmandela/child_speech_mBERT_classifier/blob/main/sampleSents.png" width="500" ></a>




