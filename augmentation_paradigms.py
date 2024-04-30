#**********************************************************************************************************************#
# # Initializing Augmentation Strategies
#**********************************************************************************************************************#

def synonym_augmentation():         # 19

    print()
    print('Synonym augmentation ++++ n augmented_paradigms.py')

    aug_syn = naw.SynonymAug(
        aug_src='wordnet',
        aug_max=3)

    return aug_syn
def contextual_embedding_augmentation():    # 19

    print()
    print('Contextual Embedding augmentation ++++ in augmented_paradigms.py')

    aug_emb = naw.ContextualWordEmbsAug(
        # Other models include 'distilbert-base-uncased', 'roberta-base', etc.
        model_path=PATH,
        # You can also choose "insert"
        action="substitute",
        # Use GPU
        device='cuda')

    return aug_emb

#**********************************************************************************************************************#
# # BACKTRANSLATION
#**********************************************************************************************************************#


# def backtranslation_augmentation_greek():
#     # Initiate the back translation augmentation
#     aug_bt = naw.BackTranslationAug(
#       # Translate English to Greek
#       to_model_name = "Helsinki-NLP/opus-mt-en-grk",
#       # Translate from Greek back to English
#       from_model_name = "Helsinki-NLP/opus-mt-grk-en",
#       # Use GPU
#       device = 'cuda')


# **********************************************************************************************************************#
# # Synonym Evaluation
# **********************************************************************************************************************#


def synonym_evaluation_score():
    # Evaluate the synonym text augmentation
    score_synonym = evaluate_aug(aug_strategy='synonym',
                                 n=2,
                                 train=train,
                                 ds_val=ds_val,
                                 ds_test=ds_test)

    print(score_synonym)
    return score_synonym

# **********************************************************************************************************************#
# # Contextual Evaluation
# **********************************************************************************************************************#


def contextual_evaluation_score():
    # Evaluate the embedding text augmentation
    score_embedding = evaluate_aug(aug_strategy='embedding',
                                   n=2,
                                   train=train,
                                   ds_val=ds_val,
                                   ds_test=ds_test)

    print(score_embedding)
    return score_embedding

# **********************************************************************************************************************#
# # BACKTRANSLATION Evaluation
# **********************************************************************************************************************#


def backtranslation_evaluation_score():
    # Evaluate the back translation text augmentation
    score_backTransLation = evaluate_aug(aug_strategy='backtranslation',
                                         n=2,
                                         train=train,
                                         ds_val=ds_val,
                                         ds_test=ds_test)

    print(score_backTransLation)
    return score_backTransLation

#**********************************************************************************************************************#
# # Evaluation of Augmentation Strategies
#**********************************************************************************************************************#

# Create a function to evaluate the text augmentation on the model performance
# on the set.

def evaluate_aug(aug_strategy, n, train, ds_val, ds_test):

    print()
    print("")
    print("Define Evaluation Strategy for Augmentation")
    print("+++++++ In augmented_paradigms.py +++++++")
    print('Evaluation Strategy for Augmentation in augmented_paradigms.py')
    print()

    global output_dir

    # Create two lists to store the augmented sentences and their corresponding labels
    augmented_text = []
    augmented_text_labels = []
    # Loop through the train set to creat augmented sentences
    # We can set the number of augmented sentences we want
    # to create per original sentence.
    for i in train.index:
        if aug_strategy == "synonym":
            lst_augment = aug_syn.augment(train['text'].loc[i], n = n)
        elif aug_strategy == 'embedding':(
            lst_augment) = aug_emb.augment(train['text'].loc[i], n = n)
        elif aug_strategy == 'backtranslation':
            lst_augment = aug_bt.augment(train['text'].loc[i], n = n)
        for augment in lst_augment:
            augmented_text.append(augment)
            augmented_text_labels.append(train['label'].loc[i])

    # Zip the two lists into a list of tuples
    augmented_text_labels = list(zip(augmented_text, augmented_text_labels))

    # Convert the list of tuples to a Pandas Dataframe.
    ds_augmented_text_labels = pd.DataFrame(
        augmented_text_labels, columns = ['text', 'label'])

    # Vertically concat the train set with the augmented texts
    train_augmented = pd.concat([train, ds_augmented_text_labels], axis = 0)

    # Convert the DataFrame to a Dataset (Aparche Arrow format)
    dset_train_augmented = Dataset.from_pandas(train_augmented)

    # Gather train, val, and test Datasets to have a single DatasetDict,
    # which can be manipulated together
    text_augmented = DatasetDict({
        'train': dset_train_augmented,
        'val': ds_val,
        'test': ds_test})
    text_augmented = text_augmented.remove_columns(["__index_level_0__"])

    # Tokenize the sentences dataset
    text_augmented_encoded = text_augmented.map(
        tokenize,
        batched = True,
        batch_size = None)

    # Remove the text column from the encoded DatasetDict because the model does not use it.
    text_augmented_encoded = text_augmented_encoded.remove_columns(['text'])

    # Since the model expects tensors as inputs,
    # we will convert the input_ids and attention_mask columns to the "torch" format.
    text_augmented_encoded.set_format(
        "torch", columns = ["input_ids", "attention_mask", "label"])

    # Define the trainer
    trainer = Trainer(
        model = model,
        # Training argument
        args = training_args,
        # Metrics (f1 score and accuracy)
        compute_metrics = compute_metrics,
        # Train and val Datasets
        train_dataset = text_augmented_encoded["train"],
        eval_dataset = text_augmented_encoded["val"],
        # Tokenizer
        tokenizer = tokenizer)

    # Clean up the memory using the garbage cleaner
    gc.collect()
    torch.cuda.empty_cache()

    # Start the training process
    trainer.train()

    # Use the model to predict the test set
    preds_output = trainer.predict(text_encoded["test"])
    print(preds_output.metrics)


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Remove all elements from the lists
    augmented_text.clear()
    augmented_text_labels.clear()

    return preds_output, augmented_text_labels

    # tokenizer.model_max_length



