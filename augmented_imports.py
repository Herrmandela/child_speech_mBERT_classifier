print()
print("imports in augmented_imports.py")
print()


# Install the tokenizer
import sacremoses

# Import the nlpaug module and its methods
import nlpaug.augmenter.word as naw
import nlpaug.flow as nafc
from nlpaug.util import Action



import os
import pandas as pd
import numpy as np
import transformers
import tensorflow as tf
import torch

from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

from sklearn.model_selection import train_test_split

from torch.utils.data import (Dataset, DataLoader,
                              RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import (BertTokenizer, BertModel, BertConfig,
                          AutoTokenizer,DataCollatorWithPadding,
                          AutoModelForSequenceClassification, Trainer,
                          TrainerCallback, TrainingArguments)

from datasets import Dataset, load_metric, DatasetDict

import matplotlib.pyplot as plt
import gc

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")