from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
import tensorflow as tf
import torch
import time
import random
import numpy as np
import pandas as pd

from functionality import train_dataloader, validation_dataloader, format_time, flat_accuracy

#**********************************************************************************************************************#
# # Define PATH
#**********************************************************************************************************************#
print()
print("Defining PATH, epochs and num_labels ++++++ in training.py")

PATH = "/data/mBERT"

epochs = 2

device = torch.device("cuda")

model = ""

preds = ""

optimizer = ""

scheduler = ""

df_stats = ""

training_stats = []

#**********************************************************************************************************************#
# # TRAINING DEPTH -  for layers training protocol  (PATH)
#**********************************************************************************************************************#

print()
print("Choosing Depth ++++++ in training.py")

def trainingVanilla():      # 7

    print()
    print('The Vanilla Model Was Chosen ++++++ in training.py')
    print()

    global PATH, model

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top

    model = BertForSequenceClassification.from_pretrained(
        PATH,                           # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2,                 # The number of output labels--2 for binary classification
                                        # - Can be increased for multiclassification tasks.
        output_attentions = False,      # Whether the model should return attention weights
        output_hidden_states = False,   # Whether the model should return all hidden states
    )

    #model.to(device)
    model.cuda()                       # Tell the model to run on GPU

    print()
    print("model: ", model)
    print()

    return model



def trainingInherent():     # 7

    print()
    print('Inherent Model Chosen ++++++ in training.py')
    print()

    global PATH, model

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top

    model = BertForSequenceClassification.from_pretrained(
        PATH,                           # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2,                 # The number of output labels--2 for binary classification
                                        # - Can be increased for multiclassification tasks.
        output_attentions = False,      # Whether the model should return attention weights
        output_hidden_states = False,   # Whether the model should return all hidden states
    )

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    ## inherent - Only freezes the input and output layers (pooler)

    list(model.parameters())[5].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                       # Tell the model to run on GPU

    return model



def trainingShallow():      # 7

    print()
    print('Shallow Model Chosen ++++++ in training.py')
    print()

    global PATH, model

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top

    model = BertForSequenceClassification.from_pretrained(
        PATH,                           # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2,                 # The number of output labels--2 for binary classification
                                        # - Can be increased for multiclassification tasks.
        output_attentions = False,      # Whether the model should return attention weights
        output_hidden_states = False,   # Whether the model should return all hidden states
    )

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    ## inherent - Only freezes the input and output layers (pooler)

    list(model.parameters())[53].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                       # Tell the model to run on GPU

    print()
    print("model: ", model)
    print()

    return model

#**********************************************************************************************************************#
# # Optimizer
#**********************************************************************************************************************#

def optimizer_and_scheduler():        # 8

    global model, optimizer, scheduler, epochs

    print()
    print("optimizer Chosen ++++++ in training.py")


    optimizer = torch.optim.AdamW(model.parameters(),
                      lr = 5e-5,      # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8)     # args.adam_epsilon  - default is 1e-8.

    print("optimizer: ", optimizer)
    print(len(optimizer.param_groups))

#**********************************************************************************************************************#
# # Linear Scheduler epochs)
#**********************************************************************************************************************#

    print("Linear scheduler START ++++++ in training.py")
    global train_dataloader
    print("******+++++++++++++*******+++++++++++++****dataloader_length: ", train_dataloader)
    print("optimizer: ", optimizer)

    epochs = int(input("Please enter the number of epochs - 4 are recommended: "))
    # The total number of training steps is the [number of batches] x [number of epochs] ** not equal to training samples!!
    total_steps = len(train_dataloader) * epochs
    print("******+++++++++++++*******+++++++++++++****total_steps: ", total_steps)
    time.sleep(5)

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # run_glue.py default value
                                                num_training_steps=total_steps)


    return optimizer, scheduler, epochs
# #**********************************************************************************************************************#
# # TRAINING  (epochs, data_loader, format_time, device)
#**********************************************************************************************************************#

def train():   # 12

    global train_dataloader, validation_dataloader, optimizer, model, scheduler, training_stats, epochs


    print("Starting training ++++++ in training.py")

    # This training code is based on the 'run_glue.py' script found here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 22

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # Validation accuracy, and timings.

    # measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print(" ")
        print("======== Epoch {:}/{:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # 'train' just flips the *mode*, it doesn't *perform* the training.
        # 'dropout' and 'batch-norm' layers behave differently during the training
        # vs 'test'(source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
          # Progress update every 40 batches.
          if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress
            print(' Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

          # Unpack this training batch from our dataloader.
          #
          # As we unpack the batch, we'll also copy each tensor  to the
          # GPU using the 'to' method.
          #
          # 'batch' contains three PyTorch tensors
          #       [0]: input ids
          #       [1]: attention masks
          #       [2]: labels
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_scores = batch[2].to(device)

          # Always clear any previously calculated gradients before preforming a
          # Backwards pass. PyTorch doesn't do this automatically because
          # accumulating the gradient is "convenient while training" RNNs.
          # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
          model.zero_grad()

          # Perform a forward pass (Evaluate the model on the training batch).
          # The documentation for this 'model' function is here:
          # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
          # They return different numbers of parameters depending on what arguments
          # are given and flags are set. For out use-case, it returns the loss
          # because we provided labels
          # -- and it also returns the 'logits' -- the model outputs prior to activation
          result = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,
                          labels=b_scores)

          loss = result.loss
          logits = result.logits

          # Accumulate the training loss over all of the batches so that we can
          # calculate the average loss at the end. 'loss' is a Tensor containing a
          # single value; the '.item()' function just returns the PyTorch value
          # from the tensor.
          total_train_loss += loss.item()

          # Perform a backward pass to calculate the gradients.
          loss.backward()

          # Clip the norm of the gradients to 1.0.
          # This is to help prevent the "exploding gradient" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          # Update parameters and take a step using the computed gradient.
          # The optimizer dictates the "update rule" --how the parameters are
          # modified based on their gradients, the learning rate, etc.
          optimizer.step()

          # Update the learning rate
          scheduler.step()

        # Calculate the average loss over all batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print(" Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
          # Unpack this validation batch from our dataloader.
          #
          # As we unpack the batch, we'll also copy each tensor  to the
          # GPU using the 'to' method.
          #
          # 'batch' contains three PyTorch tensors
          #       [0]: input ids
          #       [1]: attention masks
          #       [2]: labels
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_scores = batch[2].to(device)

          # Tell PyTorch not to bother with constructing the compute graph during
          # The forward pass, since this is only needed for back-propagation
          with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this 'model' function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the 'logits' output by the model. The 'logits' are the output
            # values prior to applying an activation function like the softmax
            result = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_scores,
                            return_dict=True)

          # Get the loss and "logits" output by the model. The "logits" are the
          # output values prior to applying an activation function like the
          # softmax.
          loss = result.loss
          logits = result.logits

          # Accumulate the validation loss.
          total_eval_loss += loss.item()

          # Move logits and labels to CPU
          logits = logits.detach().cpu().numpy()
          score_ids = b_scores.to('cpu').numpy()

          # Calculate the overall accuravy for this batch of test sentences, and
          # accumulate it over all batches.
          total_eval_accuracy += flat_accuracy(logits, score_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    #return training_stats

#**********************************************************************************************************************#
# # Training Summary
#**********************************************************************************************************************#

def training_summary():     # 13

    global df_stats

    print()
    print("Training Summary ++++++ in training.py")


    # Display floats with two decimal places.
    pd.set_option('display.precision', 2)

    # Create a DataFrame fom our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    df_stats

    return df_stats