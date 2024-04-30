print()
print()
print("|| +++In augmented_training.py +++ Loaded in augmented_three.py ||")
print()
print()

#
# global PATH
#
# tokenizer = AutoTokenizer.from_pretrained(model)

#**********************************************************************************************************************#
# # Define Tokenization Process for the Augmented-Models
#**********************************************************************************************************************#

def tokenize(batch):            # 4
    print()
    print("")
    print("Tokenization Process Function +++++++ in augmented_training.py")

    return tokenizer(
        batch["text"],
        # Pad the examples with zeros to the size of the longest one in a batch
        padding = True,
        # Truncate the examples to the model’s maximum context size (which is 512 for this model)
        truncation = True)

#**********************************************************************************************************************#
# # Text encoding function for the Augmented Models
#**********************************************************************************************************************#


#def text_encoder(sentences):
def text_encoder():             # 5

    print()
    print("")
    print("Text encoding function +++++++ in augmented_training.py")

    # Once we’ve defined a processing function, we can apply it across all the splits in the DataDict.
    text_encoded = sentences.map(
      tokenize,
      # Encode the sentences in batches
      batched = True,
      batch_size = None)

    # Apply the tokenize function on the full dataset as a single batch
    # Note: This ensures that the input tensors and attention masks have the same shape globally
    # Alternatively, we can specify max_length in the tokenize() function to ensure the same

    # Remove the text column from the encoded DatasetDict because the model does not use# it.
    text_encoded = text_encoded.remove_columns(['token_type_ids', 'text'])

    # Since the model expects tensors as inputs,
    # we will convert the input_ids and attention_mask columns to the "torch" format.
    text_encoded.set_format("torch", columns = ["input_ids", "attention_mask", "label"])

    return text_encoded

#**********************************************************************************************************************#
# # Label and ID Dictionaries for the Augmented-Models
#**********************************************************************************************************************#


def label_id_dictionaries():            # 6

    print()
    print("")
    print("Label and ID Dictionaries +++++++ in augmented_training.py")


    Define two dictionaries that convert between ids (0, 1, 2, 3) and labels (strings)
    # Note: By adding label2id and id2label to our model's config,
    # we will get friendlier labels in the inference API.
    label2id = {}
    id2label = {}
    labels = ['incorrect', 'pass', 'acceptable ', 'correct']
    for i, label_name in enumerate(labels):
        label2id[label_name] = str(i)
        id2label[str(i)] = label_name

    # Take a look the two dictionaries
    #label2id, id2label, len(label2id)

    return label2id, id2label

#**********************************************************************************************************************#
# # Define Training Depth for the Augmented-Models
#**********************************************************************************************************************#


def augmented_training_vanilla():

    print()
    print("")
    print("Define Model Parameters ++++ In augmented_training.py")
    print('The Augmented Vanilla Model Was Chosen in augmented_training.py')


    global tokenizer
    global PATH

    model = AutoModelForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        # Number of classes/labels
        num_labels=len(label2id),   # A dictionary linking label to id
        label2id=label2id,          # A dictionary linking id to label
        id2label=id2label
    )


    model.cuda()                       # Tell the model to run on GPU

    return model


def augmented_training_inherent():

    print()
    print("")
    print("Define Model Parameters ++++ In augmented_training.py")
    print('The Augmented Inherent Model was Chosen in augmented_training.py')


    global tokenizer
    global PATH

    model = AutoModelForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        # Number of classes/labels
        num_labels=len(label2id),  # A dictionary linking label to id
        label2id=label2id,  # A dictionary linking id to label
        id2label=id2label
    )

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    ## inherent - Only freezes the input and output layers (pooler)

    list(model.parameters())[5].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                       # Tell the model to run on GPU

    return model


def augmented_training_shallow():

    print()
    print("")
    print("Define Model Parameters ++++ In augmented_training.py")
    print('The Augmented Shallow Model was Chosen in augmented_training.py')

    global tokenizer
    global PATH

    model = AutoModelForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        # Number of classes/labels
        num_labels=len(label2id),  # A dictionary linking label to id
        label2id=label2id,  # A dictionary linking id to label
        id2label=id2label
    )

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    # ShallowFreeze+pooler

    list(model.parameters())[53].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                       # Tell the model to run on GPU

    return model


#**********************************************************************************************************************#
# # Define Training with LoRA
#**********************************************************************************************************************#
def lora_training_params():
    from transformers import AutoModelForTokenClassification
    from peft import PeftModel


    # instantiate base model
    model = AutoModelForTokenClassification.from_pretrained(
        llm_name,
        num_labels=13,
        id2label=id2label,
        label2id=label2id
    )
    # load LoRA model
    inference_model = PeftModel.from_pretrained(model, "lora_model_fintetuned")
    # merge and save model
    merged_model = inference_model.merge_and_unload()
    merged_model.save_pretrained("./full_finetuned_model")

#**********************************************************************************************************************#
# # CallBack for the Augmented-Models
#**********************************************************************************************************************#

class PrinterCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):   # 9
        print(f"Epoch {state.epoch}: ")

    # **********************************************************************************************************************#
    # # Define Training Arguments for the Augmented-Models
    # **********************************************************************************************************************#

    def augmented_set_training_args(self):                              # 10

        print()
        print("9")
        print("Augmented training arguments ++++ In augmented_training.py ")

        ## Define the training arguments
        output_dir = input("Please enter the output directory: ")
        # Batch size
        batch_size = 32

        # Number of epochs
        num_epochs = 4

        # Training argument
        training_args = TrainingArguments(
            # Output directory
            # Note: All model checkpoints will be saved to the folder named `model_name`
            output_dir,
            # Number of epochs
            num_train_epochs=num_epochs,
            # Learning rate
            learning_rate=2e-5,
            # Batch size for training and validation
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            # Weight decay for regularization
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=178,
            # Validate the model using the val set after each epoch
            evaluation_strategy="epoch",
            # Load the best model at the end of training
            load_best_model_at_end=True,
            # Push to Huggingface Hub
            # It could be helpful to push the model to the Hub for sharing and using pipeline(), but
            # it takes a very long time to push the model, so we choose not do it here.
            push_to_hub=False,
            # Save model checkpoint after each epoch
            save_strategy="epoch")

        # Define the trainer

        trainer = Trainer(
            # Model
            model=model,
            # Training argument
            args=training_args,
            # Metrics (f1 score and accuracy)
            compute_metrics=compute_metrics,
            # Train and val Datasets
            train_dataset=text_encoded["train"],
            eval_dataset=text_encoded["val"],
            # Tokenizer
            tokenizer=tokenizer)

        print("8")
        print("Garbage Collector to clean Memory ++++++ in augmented_training.py")

        gc.collect()
        torch.cuda.empty_cache()

        print("Trainer.train() ++++++ in augmented_training.py")
        print("8")

        trainer.train()

        # trainer.state.log_history

        # **********************************************************************************************************************#
        # # Defining metrics
        # **********************************************************************************************************************#

        print("9")
        print("Loss, F1 and Accuracy Scores ++++ In augmented_metrics.py")


        train_loss = []
        for elem in trainer.state.log_history:
            if 'loss' in elem.keys():
                train_loss.append(elem['loss'])

        Macro_f1 = []
        for elem in trainer.state.log_history:
            if 'eval_f1' in elem.keys():
                Macro_f1.append(elem['eval_f1'])

        accuracy = []
        for elem in trainer.state.log_history:
            if 'eval_accuracy' in elem.keys():
                accuracy.append(elem['eval_accuracy'])

        val_loss = []
        for elem in trainer.state.log_history:
            if 'eval_loss' in elem.keys():
                val_loss.append(elem['eval_loss'])

        return train_loss, Macro_f1, accuracy, val_loss

def synonym_augmented_training():

    print()
    print("")
    print("synonym_augmented_training")
    print("+++++++ In augmented_metrics.py +++++++")
    print()

    # Evaluate the synonym text augmentation
    score_synonym = evaluate_aug(
      aug_strategy = 'synonym',
      n = 2,
      train = train,
      ds_val = ds_val,
      ds_test = ds_test)
    print(score_synonym)


    from functionality import augmented_load_model
    augmented_load_model(output_dir)

    from mcc_evaluation import augmented_mcc_evaluation
    augmented_mcc_evaluation()

    from data_visualization import augmented_mccPlot, augmented_confusion_matrix
    augmented_mccPlot()
    augmented_confusion_matrix()

def contextual_embedding_augmented_training():

    print()
    print("")
    print("contextual_embedding_augmented_training")
    print("+++++++ In augmented_metrics.py +++++++")
    print()

    # # Evaluate the embedding text augmentation
    # score_embedding = evaluate_aug(
    #     aug_strategy='embedding',
    #     n=2,
    #     train=train,
    #     ds_val=ds_val,
    #     ds_test=ds_test)
    # print(score_embedding)


    # from functionality import augmented_load_model
    # augmented_load_model(output_dir)
    #
    # from mcc_evaluation import augmented_mcc_evaluation
    # augmented_mcc_evaluation()
    #
    # from data_visualization import augmented_mccPlot, augmented_confusion_matrix
    # augmented_mccPlot()
    # augmented_confusion_matrix()
