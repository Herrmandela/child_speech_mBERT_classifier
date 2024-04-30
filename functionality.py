from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import datetime


batch_size = 32

model = ""
tokenizer = ""


#**********************************************************************************************************************#
# # Training and Validation Splits
#**********************************************************************************************************************#

def trainingAndValidation():

    print()
    print("training and validation in functionality.py")

    global input_ids, attention_mask, scores

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, scores)

    # Creat a 90-10 train-validation split and calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #print('{:>5,} training samples'.format(train_size))
    #print('{:>5,} validation samples'.format(val_size))

    return train_dataset, val_dataset



#**********************************************************************************************************************#
# # Memory Iteration - Batch_Size
#**********************************************************************************************************************#

def saveMemory(batch_size):

    print()
    print("Saving memory in functionality.py")


    # The DataLoader needs to know our batch-size for training, so we specify it here.
    # For fine-tuning BERT on a specific task, the authors recommend the size of 16 or 32.

    # Create the DataLoaders for our training and validation sets.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with the set Batch size
    )

    # For the validation the order doesn't matter, so they'll be read sequentially
    validation_dataloader = DataLoader(
        val_dataset,  # Validation samples
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size  # evaluate with the set Batch size
    )

    return train_dataloader, validation_dataloader

#**********************************************************************************************************************#
# #  Accuracy
#**********************************************************************************************************************#

# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, scores):   # 10

    print()
    print("FlatAcc in Functionality.py")

    pred_flat = np.argmax(preds, axis=1).flatten()
    scores_flat = scores.flatten()
    return np.sum(pred_flat == scores_flat)/ len(scores_flat)


#**********************************************************************************************************************#
# # Elapsed Timer
#**********************************************************************************************************************#
def format_time(elapsed):            # 11

    print()
    print("format time in functionality.py")

    # time in seconds is returned as a string hh:mm:ss

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds = elapsed_rounded))


#**********************************************************************************************************************#
# #  SAVE Model  and Tokenizer
#**********************************************************************************************************************#

def save_model():                   # 18

    print()
    print("Save Model in functionality.py")

    output_dir = input("Please enter the output directory: ")

    #global output_dir


    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    return output_dir     #,tokenizer




#**********************************************************************************************************************#
# #  LOAD Model  and Tokenizer      (model, tokenizer, device)
#**********************************************************************************************************************#
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(output_dir):         # 19

    print("output_dir is %s" % output_dir)


    print()
    print("Loading model in functionality.py" )


    id2label = {0: "INCORRECT", 1: "CORRECT"}

    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForSequenceClassification.from_pretrained(output_dir, num_labels=2, id2label=id2label)

    model = model_class.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Copy the model to the GPU.
    model.to(device)
    print(output_dir)

    return model, tokenizer


#**********************************************************************************************************************#
# #  Multiclass save Model and Tokenizer
#**********************************************************************************************************************#

def multiclass_save_model():    # 22

    import os

    print()
    print("Multiclass save Model in functionality.py")
    print("20")
    print()

    output_dir = input("Please enter the output directory: ")
    # global output_dir

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()


    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

#**********************************************************************************************************************#
# #  Multiclass Load model and Tokenizer
#**********************************************************************************************************************#

def multiclass_load_model(output_dir):          # 23
#     from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("output_dir is %s" % output_dir)

    print()
    print("Multiclass load Model in functionality.py")
    print("21")
    print()

    id2label = {0: "INCORRECT", 1: "CORRECT"}

    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForSequenceClassification.from_pretrained(output_dir,
                                                               num_labels=2,
                                                               id2label=id2label)

    model = model_class.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Copy the model to the GPU.
    model.to(device)

#**********************************************************************************************************************#
# #  Augmented save Model and Tokenizer
#**********************************************************************************************************************#


def augmented_save_model():            # 14

    print()
    print("Augmented save Model in functionality.py")
    print("")
    print()


    output_dir = input("Please enter the output directory for your model and tokinzer: ")

    '''
    Since we specified the option "load_best_model_at_end = True" in the training argument,
    the current model is the best one. We now want to save the model locally.
    Note: We can still use the pre-trained model's tokenizer since we did not modify it,
    but for illustration purpose, we save the tokenizer as well.
    '''

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


#**********************************************************************************************************************#
# #  Augmented load model and Tokenizer
#**********************************************************************************************************************#


def augmented_load_model(output_dir):       # 15

    print()
    print("Augmented load Model in functionality.py")
    print("")
    print()

    print("output_dir is %s" % output_dir)

    classifier = pipeline(
        "text-classification",
        model=os.path.join(output_dir),
        tokenizer=os.path.join(output_dir))

#**********************************************************************************************************************#
# #  Augmented test English
#**********************************************************************************************************************#


def augmented_eng_test():

    print()
    print("Testing Augmented Model in functionality.py")
    print("")
    print()

    # # Then we can test the pipeline with a sample sentence (or a sentence from our test set).
    # English_text = "The parent cooked a tasty dish."
    #
    # preds_English = classifier(English_text, top_k=None)
    #
    # print('\nThe predictions for the sentence "The parent cooked a tasty dish." are:\n')
    # preds_English

