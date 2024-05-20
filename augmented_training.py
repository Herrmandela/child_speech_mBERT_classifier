import torch
from transformers import AutoTokenizer, BertTokenizer, Trainer, TrainerCallback, TrainingArguments, AutoModelForSequenceClassification
import gc
import numpy as np

from augmented_data_prep import text_encoded
from augmented_metrics import compute_metrics

# Set the model path
PATH = "/mBERT"

# Define the tokenizer
#tokenizer = AutoTokenizer.from_pretrained(PATH)

tokenizer = BertTokenizer.from_pretrained(PATH)
#**********************************************************************************************************************#
# # Label and ID Dictionaries for the Augmented-Models
#**********************************************************************************************************************#

device = torch.device("cuda")

# Define two dictionaries that convert between ids (0, 1, 2, 3) and labels (strings)
# Note: By adding label2id and id2label to our model's config,
# we will get friendlier labels in the inference API.
label2id = {}
id2label = {}
labels = ['incorrect', 'pass', 'acceptable ', 'correct']
for i, label_name in enumerate(labels):
    label2id[label_name] = str(i)
    id2label[str(i)] = label_name

model = AutoModelForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        # Number of classes/labels
        num_labels=len(label2id),   # A dictionary linking label to id
        label2id=label2id,          # A dictionary linking id to label
        id2label=id2label
    )


# Take a look the two dictionaries
# label2id, id2label, len(label2id)

# return label2id, id2label

#**********************************************************************************************************************#
# # Define Training Depth for the Augmented-Models
#**********************************************************************************************************************#


def augmented_training_vanilla():

    print()
    print("")
    print("Define Model Parameters ++++ In augmented_training.py")
    print('The Augmented Vanilla Model Was Chosen in augmented_training.py')

    global tokenizer, PATH, model, device

    model.cuda()                        # Tell the model to run on GPU

    print("model : ", model, "type : ", type(model))
    return model


def augmented_training_inherent():

    print()
    print("")
    print("Define Model Parameters ++++ In augmented_training.py")
    print('The Augmented Inherent Model was Chosen in augmented_training.py')

    global tokenizer, PATH, model

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

    global tokenizer, PATH, model


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
"""
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
    merged_model.save_pretrained("./full_finetuned_model")"""

# **********************************************************************************************************************#
# # Define Training Arguments and CallBack for the Augmented-Models
# **********************************************************************************************************************#


print()
print("9")
print("Augmented training arguments ++++ In augmented_training.py ")

output_dir = input("Please enter the output directory: ")
# Batch size
batch_size = 32

max_length = 28

# Number of epochs
num_epochs = int(input("Please enter the number of epochs: (We recommend 3-4)"))

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

trainer.state.log_history

class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")

label2id, id2label, len(label2id)

# **********************************************************************************************************************#
# # Defining metrics
# **********************************************************************************************************************#

preds_output = trainer.predict(text_encoded["val"])

preds_output.metrics

y_preds = np.argmax(preds_output.predictions, axis = 1)

# Model prediction metrics on the test set
preds_output = trainer.predict(text_encoded["test"])
preds_output.metrics


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