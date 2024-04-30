print()
print()
print("|| +++ In augmented_data_prep.py +++ Loaded ||")

# **********************************************************************************************************************#
# SPLIT MULTICLASS DATA INTO TRAINING, VALIDATION AND TESTING DATA - *****
# **********************************************************************************************************************#

def augmented_data_split():

    print()
    print("5")
    print("Training, Validation and Testing Splits ++++++ in augmented_data_prep.py")


    # split bulk data into train and test set
    train, test = train_test_split(df, test_size=0.2, random_state=23)

    # split the training into validation and test set
    val, rest = train_test_split(All_text, test_size=0.918, random_state=23)

    # save the data
    train.to_csv('ESrtTrain.csv', index=False)
    val.to_csv('ESrtVal.csv', index=False)
    test.to_csv('ESrtTest.csv', index=False)

    train.shape, val.shape, test.shape

    # train_ds = pd.read_csv("/content/ESrtTrain.csv").astype(str)
    # val_ds = pd.read_csv("/content/ESrtVal.csv").astype(str)
    # test_ds = pd.read_csv("/content/ESrtTest.csv").astype(str)


    ds_train = Dataset.from_pandas(train)
    ds_val = Dataset.from_pandas(val)
    ds_test = Dataset.from_pandas(test)

    # Gather train, val, and test Datasets to have a single DatasetDict, and make it manipulatable
    sentences = DatasetDict({
        'train': ds_train,
        'val': ds_val,
        'test': ds_test})

    # Dataset.from_pandas will add an index column, which can be removed
    sentences = sentences.remove_columns(["__index_level_0__"])

    return sentences

