print()
print()
print("|| +++ In multiclass_data_prep.py +++ Loaded ||")
print()
print()

# from sklearn.model_selection import train_test_split
# from datasets import Dataset
#
#

# **********************************************************************************************************************#
# SPLIT MULTICLASS DATA INTO TRAINING, VALIDATION AND TESTING DATA - *****
# **********************************************************************************************************************#

def multiclass_data_split():  # 4
#    from sklearn.model_selection import train_test_split

    global df

    print()
    print("4")
    print("Training, Validation and Testing Splits ++++++ in multiclass_data_prep.py")

    #split bulk data into train and test set
    train,test = train_test_split(df, test_size=0.30, random_state=0)

    #split the training into validation and test set
    val,test = train_test_split(test, test_size=0.50, random_state=0)

    #save the data
    train.to_csv('dfTrain.csv',index=False)
    test.to_csv('dfTest.csv',index=False)
    val.to_csv('dfVal.csv', index=False)

    return train,test,val


def multiclass_data_pandafication():   # 5

    print()
    print("5")
    print("Pandafy - Training, Validation and Testing Splits ++++++ in multiclass_data_prep.py")
    print()
    
    train_ds = pd.read_csv("/content/dfTrain.csv").astype(str)
    val_ds = pd.read_csv("/content/dfVal.csv").astype(str)
    test_ds = pd.read_csv("/content/dfTest.csv").astype(str)

    train_ds = Dataset.from_pandas(train_ds)
    val_ds = Dataset.from_pandas(val_ds)
    test_ds = Dataset.from_pandas(test_ds)

    return train_ds,val_ds,test_ds



def preprocess_function(examples: dict):      # 6

    print()
    print("6")
    print("Preprocessing Function ++++++ in multiclass_data_prep.py")


    labels = [0] * len(id2label)
    for k, l in id2label.items():
        if str(l) == examples['STRUCTURE'] or str(l) in examples['MLC_CELF'] or str(l) in examples['MLC_TOLD']:
            labels[k] = 1
        else:
            labels[k] = 0
    examples = tokenizer(examples["RESPONSE"], padding="max_length", max_length=44)
    examples['labels'] = labels
    return examples


def dataset_test_evaluation_english():      # 7

    print()
    print("7")
    print("Split Mapping English ++++++ in multiclass_data_prep.py")
    print()
    # This function displays the examples that the preprocessing function maps out.

    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}

    for split in ds:
        ds[split] = ds[split].map(preprocess_function,
                                  remove_columns=['ORDER', 'CODE', 'STRUCTURE', 'ITEM',
                                                  'TARGET', 'RESPONSE', 'Error_Types', 'AUTO_SCORING',
                                                  'CELF_SCORING', 'TOLD_SCORING','GRAMMATICAL',
                                                  'STRUCTURE_SCORE', 'GENDER', 'AGE', 'AGE_of_ONSET_EN',
                                                  'AGE_of_ONSET_other', 'AoO_bi', 'ENGL_STATUS',
                                                  'OTHER_LANG', 'TOTAL_TOLD', 'TOTDAL_CELF',
                                                  'TOTAL_STRUCTURE', 'TOTAL_CELF', 'SUBGROUP',
                                                  'MLC_CELF', 'MLC_TOLD'])



def dataset_test_evaluation_farsi():        # 7

    print()
    print("7")
    print("Split Mapping Farsi ++++++ in multiclass_data_prep.py")
    print()
    # This function displays the examples that the preprocessing function maps out.

    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}

    for split in ds:
        ds[split] = ds[split].map(preprocess_function,
                                  remove_columns=['ChildID', 'STRUCTURE', 'Test Number',
                                                  'TARGET', 'RESPONSE', 'RESPONSE', 'Auto Scoring',
                                                  'TOLD_SCORING', 'CELF_SCORING',
                                                  'MLC_TOLD', 'MLC_CELF'], )




def dataset_test_evaluation_greek():      # 7

    print()
    print("7")
    print("Split Mapping Greek ++++++ in multiclass_data_prep.py")
    print()
    # This function displays the examples that the preprocessing function maps out.

    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}

    for split in ds:
        ds[split] = ds[split].map(preprocess_function,
                                  remove_columns=['ChildID', 'STRUCTURE', 'TEST_NUMBER',
                                                  'TARGET', 'RESPONSE', 'AUTO_SCORING', 'CELF_SCORING',
                                                  'TOLD_SCORING', 'MLC_CELF', 'MLC_TOLD'])
