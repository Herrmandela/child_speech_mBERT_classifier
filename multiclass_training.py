print()
print()
print("|| +++In multiclass_training.py +++ Loaded in multiclass_two.py ||")

#**********************************************************************************************************************#
# # Define Training Parameters for the MultiClass-Models
#**********************************************************************************************************************#

def training_parameters():          # 2

    print()
    print("2")
    print("Define Training Parameters +++++++ in multiclass_training.py")


    LEARNING_RATE = 2e-4
    MAX_LENGTH = 44
    BATCH_SIZE = 32
    EPOCHS = 4

    """
    #New
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 44
    BATCH_SIZE = 32
    EPOCHS = 20
    """

    # Name the classes
    id2label = {k: l for k, l in enumerate(ALL_LABELS)}
    label2id = {l: k for k, l in enumerate(ALL_LABELS)}

    return (LEARNING_RATE, MAX_LENGTH,
            BATCH_SIZE, EPOCHS,
            label2id, id2label)

    #Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top

#**********************************************************************************************************************#
# # MODEL TRAINING DEPTH -  for layers training protocol
#**********************************************************************************************************************#

def multiclass_training_vanilla():          # 3

    print()
    print("3")
    print("Define Model Parameters")
    print('The MultiClassification Vanilla Model Was Chosen ++++ in multiclass_training.py')


    global tokenizer
    global PATH

    # tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        id2label=id2label,  # A dictionary linking label to id
        label2id=label2id,  # A dictionary linking id to label
    )

    model.cuda()                       # Tell the model to run on GPU

    return model


def multiclass_training_inherent():         # 3

    print()
    print("3")
    print("Define Model Parameters")
    print('The MultiClassification Inherent Model was Chosen ++++ in multiclass_training.py')
    print()

    global tokenizer
    global PATH

    # tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        PATH,                           # Use the 12-layer BERT model, with an uncased vocab.
        id2label=id2label,  # A dictionary linking label to id
        label2id=label2id,  # A dictionary linking id to label
    )

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    ## inherent - Only freezes the input and output layers (pooler)

    list(model.parameters())[5].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                       # Tell the model to run on GPU

    return model


def multiclass_training_shallow():          # 3

    print()
    print("3")
    print("Define Model Parameters")
    print('The MultiClassification Shallow Model was Chosen ++++ in multiclass_training.py')


    global tokenizer
    global PATH

    # tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        id2label=id2label,  # A dictionary linking label to id
        label2id=label2id,  # A dictionary linking id to label
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
# # Optimizer
#**********************************************************************************************************************#


#**********************************************************************************************************************#
# # Linear Scheduler (epochs, train_dataloader)
#**********************************************************************************************************************#


# #**********************************************************************************************************************#
# # TRAINING    (epochs, data_loader, format_time, device)
#**********************************************************************************************************************#

class MultiTaskClassificationTrainer(Trainer):         # 10

    print()
    print("10")
    print("Class MultiTaskClassificationTrainer(Trainer) ++++ In multiclass_training.py")

    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights

    def compute_loss(self, model, inputs, return_outputs=False): # 11

        print()
        print("11")
        print("Compute loss ++++ In multiclass_training.py ")
        print()


        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]

        structure_loss = torch.nn.functional.cross_entropy(logits[:, STRUCTURE_INDICES],
                                                           labels[:, STRUCTURE_INDICES].float())

        TOLD_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, TOLD_SCORING_INDICES],
                                                                         labels[:, TOLD_SCORING_INDICES].float())

        CELF_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, CELF_SCORING_INDICES],
                                                                         labels[:, CELF_SCORING_INDICES].float())

        loss = (self.group_weights[0] * structure_loss +
                self.group_weights[2] * TOLD_loss +
                self.group_weights[1] * CELF_loss)

        return (loss, outputs) if return_outputs else loss

class PrinterCallback(TrainerCallback):    # 12

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):   # 13

        print("+++++++ In multiclass_training.py +++++++")
        print("PrinterCallback_Step")

        print(f"Epoch {state.epoch}: ")



    def multiclass_set_training_args(self):       # 14
        output_dir = input("Please enter the output directory: ")

        training_args = TrainingArguments(
            output_dir,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            metric_for_best_model="f1_macro",
            load_best_model_at_end=True,
            weight_decay=0.01,
        )

        trainer = MultiTaskClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            compute_metrics=compute_metrics,
            callbacks=[PrinterCallback],
            group_weights=(0.7, 4, 4)
        )

        print()
        print("15")
        print("Trainer.train() ++++++ in multiclass_training.py")
        print()

        trainer.train()     # 15

        print()
        print("16")
        print("Trainer.evaluate() ++++++  in multiclass_training.py")
        print()

        trainer.evaluate()      # 16

        preds_output = trainer.predict(text_encoded["val"])
        preds_output.metrics
        y_preds = np.argmax(preds_output.predictions, axis=1)
        preds_output = trainer.predict(text_encoded["test"])
        preds_output.metrics

        print()
        print("16")
        print("returning trainer, y_preds, preds_output, in multiclass_training.py")
        print()

        return (trainer, y_preds, preds_output)    # 16

#**********************************************************************************************************************#
# # Training Summary
#**********************************************************************************************************************#

def training_summary():

    print()
    print("Training Summary ++++++  in multiclass_training.py!")
    print(25)
    #
    # # Display floats with two decimal places.
    # pd.set_option('display.precision', 2)
    #
    # # Create a DataFrame fom our training statistics.
    # df_stats = pd.DataFrame(data=training_stats)
    #
    # # Use the 'epoch' as the row index.
    # df_stats = df_stats.set_index('epoch')
    #
    # # A hack to force the column headers to wrap.
    # # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    #
    # # Display the table.
    # df_stats
    #
    # return df_stats