from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from datasets import Dataset,load_metric, DatasetDict
from transformers import AutoTokenizer

from augmented_load_data import df, All_text

# Set the model path
PATH = "/content/drive/MyDrive/data/LBERT"

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH)

sentences = []

print()
print()
print("|| +++ In augmented_data_prep.py +++ Loaded ||")

# **********************************************************************************************************************#
# SPLIT MULTICLASS DATA INTO TRAINING, VALIDATION AND TESTING DATA - *****
# **********************************************************************************************************************#

# split bulk data into train and test set
train, test = train_test_split(df, test_size=0.2, random_state=23)

# split the training into validation and test set
val, rest = train_test_split(All_text, test_size=0.918, random_state=23)

#save the data
train.to_csv('dfTrain.csv',index=False)
test.to_csv('dfTest.csv',index=False)
val.to_csv('dfVal.csv', index=False)


train.shape, val.shape, test.shape

#************

train_ds = pd.read_csv("/content/dfTrain.csv").astype(str)
val_ds = pd.read_csv("/content/dfVal.csv").astype(str)
test_ds = pd.read_csv("/content/dfTest.csv").astype(str)

print("val_ds : ", type(val_ds))
#************

ds_train = Dataset.from_pandas(train)
ds_val = Dataset.from_pandas(val)
ds_test = Dataset.from_pandas(test)

print("ds_val : ", type(ds_val))
#************

# Gather train, val, and test Datasets to have a single DatasetDict, and make it manipulatable
sentences = DatasetDict({
    'train': ds_train,
    'val': ds_val,
    'test': ds_val})

# Dataset.from_pandas will add an index column, which can be removed
sentences = sentences.remove_columns(["__index_level_0__"])


#**********************************************************************************************************************#
# # Define Tokenization Process for the Augmented-Models
#**********************************************************************************************************************#
# Define a function to compute two metrics--accuracy and f1 score
def compute_metrics(pred):
  # True labels
  labels = pred.label_ids

  preds = pred.predictions.argmax(-1)
  # Note: average = "weighted" will weigh the f1_score by class sample size
  f1 = f1_score(labels, preds, average = "weighted")
  acc = accuracy_score(labels, preds)
  # Note: Need to return a dictionary
  return {"accuracy": acc, "f1": f1}

def tokenize(batch):            # 4
    
    global PATH, tokenizer 

    return tokenizer(
        batch["text"],
        # Pad the examples with zeros to the size of the longest one in a batch
        padding = True, 
        # Truncate the examples to the model’s maximum context size (which is 512 for this model)
        truncation = True)

# Once we’ve defined a processing function, we can apply it across all the splits in the DataDict.
text_encoded = sentences.map(tokenize, batched=True, batch_size=None)

    # Apply the tokenize function on the full dataset as a single batch
    # Note: This ensures that the input tensors and attention masks have the same shape globally
    # Alternatively, we can specify max_length in the tokenize() function to ensure the same

    # Remove the text column from the encoded DatasetDict because the model does not use# it.
text_encoded = text_encoded.remove_columns(['token_type_ids', 'text'])

    # Since the model expects tensors as inputs,
    # we will convert the input_ids and attention_mask columns to the "torch" format.
text_encoded.set_format("torch", columns = ["input_ids", "attention_mask", "label"])

"""
print("text_encoded : ", text_encoded)
print()
print("sentences: ", type(sentences))
print(sentences['test'])"""
