print()
print()
print("|| +++ In multiclass_data_prep.py +++ Loaded ||")

import pandas as pd
from transformers import  AutoTokenizer, BertTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric, DatasetDict

from multiclass_load_data import df, ALL_LABELS # X


# # Define PATH
PATH = "/content/drive/MyDrive/data/mBERT"

#PATH = "/Users/ph4533/Desktop/PyN4N/Py38/gn4n/mBERT"

# # Define the mBERT Tokenizer
tokenizer = BertTokenizer.from_pretrained(PATH, do_lower_case=True)


id2label = {k: l for k, l in enumerate(ALL_LABELS)}
label2id = {l: k for k, l in enumerate(ALL_LABELS)}


# **********************************************************************************************************************#
# SPLIT MULTICLASS DATA INTO TRAINING, VALIDATION AND TESTING DATA - *****
# **********************************************************************************************************************#


#split bulk data into train and test set
train,test = train_test_split(df, test_size=0.30, random_state=0)

#split the training into validation and test set
val,test = train_test_split(test, test_size=0.50, random_state=0)

#save the data
train.to_csv('dfTrain.csv',index=False)
test.to_csv('dfTest.csv',index=False)
val.to_csv('dfVal.csv', index=False)

train.shape, val.shape, test.shape


#************def multiclass_data_pandafication():   # 5


train_ds = pd.read_csv("/content/dfTrain.csv").astype(str)
val_ds = pd.read_csv("/content/dfVal.csv").astype(str)
test_ds = pd.read_csv("/content/dfTest.csv").astype(str)

#************

ds_train = Dataset.from_pandas(train_ds)
ds_val = Dataset.from_pandas(val_ds)
ds_test = Dataset.from_pandas(test_ds)

ds = {"train": ds_train, "validation": ds_val, "test": ds_test}

senteses = DatasetDict({
    'train': ds_train,
    'val': ds_val,
    'test': ds_val})

# Dataset.from_pandas will add an index column, which can be removed
#senteses = senteses.remove_columns(["__index_level_0__"])
# **********************************************************************************************************************#
# PREPROCESSING FUNCTION *****
# **********************************************************************************************************************#


def preprocess_function(examples: dict):      # 6

    labels = [0] * len(id2label)
    for k, l in id2label.items():
        if str(l) == examples['STRUCTURE'] or str(l) in examples['MLC_CELF'] or str(l) in examples['MLC_TOLD']:
            labels[k] = 1
        else:
            labels[k] = 0
    examples = tokenizer(examples["RESPONSE"], padding="max_length", max_length=44)
    examples['labels'] = labels
    return examples


preprocess_function(ds_test[101])


# **********************************************************************************************************************#
# DATASET SPLIT MAPPING *****
# **********************************************************************************************************************#


def dataset_test_evaluation_english():      # 7

    global ds

    print("7")
    print("Split Mapping English ++++++ in multiclass_data_prep.py")

    print(type(ds))
    print("ds in data_test_evaluation_english ", ds)
    for split in ds:
        ds[split] = ds[split].map(preprocess_function)
        """,
                                  remove_columns=['ORDER', 'CODE', 'STRUCTURE', 'ITEM',
                                                  'TARGET', 'RESPONSE', 'Error_Types', 'AUTO_SCORING',
                                                  'CELF_SCORING', 'TOLD_SCORING','GRAMMATICAL',
                                                  'STRUCTURE_SCORE', 'GENDER', 'AGE', 'AGE_of_ONSET_EN',
                                                  'AGE_of_ONSET_other', 'AoO_bi', 'ENGL_STATUS',
                                                  'OTHER_LANG', 'TOTAL_TOLD', 'TOTDAL_CELF',
                                                  'TOTAL_STRUCTURE', 'TOTAL_CELF', 'SUBGROUP',
                                                  'MLC_CELF', 'MLC_TOLD'])"""

    print("ds: ", ds)
    return ds

# **********************************************************************************************************************#

def dataset_test_evaluation_farsi():        # 7

    global ds

    print("7")
    print("Split Mapping Farsi ++++++ in multiclass_data_prep.py")

    for split in ds:
        ds[split] = ds[split].map(preprocess_function)
        """,
                                  remove_columns=['ChildID', 'STRUCTURE', 'Test Number',
                                                  'TARGET', 'RESPONSE', 'RESPONSE', 'Auto Scoring',
                                                  'TOLD_SCORING', 'CELF_SCORING',
                                                  'MLC_TOLD', 'MLC_CELF'], )"""

    print("ds: ", ds)
    return ds

# **********************************************************************************************************************#

def dataset_test_evaluation_greek():      # 7

    global ds

    print("7")
    print("Split Mapping Greek ++++++ in multiclass_data_prep.py")

    for split in ds:
        ds[split] = ds[split].map(preprocess_function)
        """
                                  remove_columns=['ChildID', 'STRUCTURE', 'TEST_NUMBER',
                                                  'TARGET', 'RESPONSE', 'AUTO_SCORING', 'CELF_SCORING',
                                                  'TOLD_SCORING', 'MLC_CELF', 'MLC_TOLD'])"""

    print("ds: ", ds)
    return ds


def tokenize(batch):  # 4

    PATH = "/content/drive/MyDrive/data/mBERT"

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PATH)

    return tokenizer(
        batch["RESPONSE"],
        # Pad the examples with zeros to the size of the longest one in a batch
        padding=True,
        # Truncate the examples to the model’s maximum context size (which is 512 for this model)
        truncation=True)

# Once we’ve defined a processing function, we can apply it across all the splits in the DataDict.
texten = senteses.map(tokenize, batched=True, batch_size=None)

    # Apply the tokenize function on the full dataset as a single batch
    # Note: This ensures that the input tensors and attention masks have the same shape globally
    # Alternatively, we can specify max_length in the tokenize() function to ensure the same

    # Remove the text column from the encoded DatasetDict because the model does not use# it.
texten = texten.remove_columns(['token_type_ids'])

    # Since the model expects tensors as inputs,
    # we will convert the input_ids and attention_mask columns to the "torch" format.
texten.set_format("torch", columns = ["input_ids", "attention_mask", "CELF_SCORING"])

