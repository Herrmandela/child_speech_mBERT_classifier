#**********************************************************************************************************************#
# # Pre_trained Model makes predictions on the multilingual Testset  (model, prediction_dataloader, device)
#**********************************************************************************************************************#
# Create sentence and label lists
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from sklearn.metrics import matthews_corrcoef
import numpy as np

mcc = 0

matthews_set = []
# *********************************************************************************************************************#
# # Binary # MCC EVALUATION
# *********************************************************************************************************************#

def binary_mcc_evaluation():    # 16 in mcc_evaluation.py
                                    # 20 for multiclass Models

    from binary_load_data import df
    from training import model
    import torch
    from model_tokenizer import tokenizer
    device = torch.device("cuda")

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    print()
    print("Binary or multiclass - MCC Evaluation in mcc_evaluation.py")

    # Tokenize all sentences and map the tokens to their word IDs

    input_ids = []
    attention_masks = []

    # For every sentence...
    for response in responses:
        # Encode_plus will...
        # 1) Tokenize the sentence
        # 2) Prepend the '[CLS]' token to the start.
        # 3) Append the '[SEP]' token to the send.
        # 4) Map the tokens ti their IDs.
        # 5) Pad or truncate the sentence to 'max_length'
        # 6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            response,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=44,  # Pad & truncate all sentences.--> FOR GREEK USE >44
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # Attention_masks differentiate relevant information from padding
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    scores = torch.tensor(scores)

    # Set the batch size
    batch_size = 256

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, scores)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Predicitons on the Test Set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_scores = [], []

    print("device: ",device)

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_scores = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        score_ids = b_scores.to('cpu').numpy()

        # remove nan from score_ids
        # score_ids = score_ids[~np.isnan(score_ids)]

        # Store predictions and true labels
        predictions.append(logits)
        true_scores.append(score_ids)

    print('    DONE.')


    print('Positive samples: %d of %d (%.2f%%)' % (df.TOLD_SCORING.sum(),
                                                   len(df.TOLD_SCORING),
                                                   (df.TOLD_SCORING.sum() /
                                                    len(df.TOLD_SCORING) * 100.0)))



    #******************************************************************************************************************#
    # Augmented # THE MCC-SET
    #******************************************************************************************************************#

    global matthews_set, mcc

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating MCC for each batch...')

    # For each batch...
    for i in range(len(true_scores)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_scores_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        # if true_scores[i] != "nan"

        matthews = matthews_corrcoef(true_scores[i], pred_scores_i)
        matthews_set.append(matthews)

    print('DONE')

    # #********************************************************************************************************************#
    # # # Final Score
    # #********************************************************************************************************************#

    # Combine all results across all batches
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score,
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_scores = np.concatenate(true_scores, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_scores, flat_predictions)

    print('Total Binary MCC: %.3f' % mcc)

    return flat_predictions, flat_true_scores, mcc, matthews_set



# *********************************************************************************************************************#
# # MultiClass # MCC EVALUATION
# *********************************************************************************************************************#


def multiclass_mcc_evaluation():    # 16 in mcc_evaluation.py
                                    # 20 for multiclass Models

    from multiclass_load_data import All_text
    from multiclass_training import model, tokenizer
    responses = All_text.RESPONSE.values
    scores = All_text.CELF_SCORING.values

    device = torch.device("cuda")

    print()
    print("Binary or multiclass - MCC Evaluation in mcc_evaluation.py")

    # Tokenize all sentences and map the tokens to their word IDs

    input_ids = []
    attention_masks = []

    # For every sentence...
    for response in responses:
        # Encode_plus will...
        # 1) Tokenize the sentence
        # 2) Prepend the '[CLS]' token to the start.
        # 3) Append the '[SEP]' token to the send.
        # 4) Map the tokens ti their IDs.
        # 5) Pad or truncate the sentence to 'max_length'
        # 6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            response,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=44,  # Pad & truncate all sentences.--> FOR GREEK USE >44
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # Attention_masks differentiate relevant information from padding
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    scores = torch.tensor(scores)

    # Set the batch size
    batch_size = 256

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, scores)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Predicitons on the Test Set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_scores = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_scores = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        score_ids = b_scores.to('cpu').numpy()

        # remove nan from score_ids
        # score_ids = score_ids[~np.isnan(score_ids)]

        # Store predictions and true labels
        predictions.append(logits)
        true_scores.append(score_ids)

    print('    DONE.')


    print('Positive samples: %d of %d (%.2f%%)' % (df.CELF_SCORING.sum(),
                                                   len(df.CELF_SCORING),
                                                   (df.CELF_SCORING.sum() /
                                                    len(df.CELF_SCORING) * 100.0)))

    # *********************************************************************************************************************#
    # # MultiClass - THE MCC
    # *********************************************************************************************************************#

    # from sklearn.metrics import matthews_corrcoef

    print()
    print("multiclass for each batch in mcc_evaluation.py")

    #global matthews_set

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating MCC for each batch...')

    # For each batch...
    for i in range(len(true_scores)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_scores_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        # if true_scores[i] != "nan"

        matthews = matthews_corrcoef(true_scores[i], pred_scores_i)
        matthews_set.append(matthews)

    print('DONE')

    # #********************************************************************************************************************#
    # # # Final Score
    # #********************************************************************************************************************#

    # Combine all results across all batches
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score,
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_scores = np.concatenate(true_scores, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_scores, flat_predictions)

    print('Total MCC: %.3f' % mcc)

    return flat_predictions, flat_true_scores, mcc, matthews_set


# *********************************************************************************************************************#
# # Augmented # MCC EVALUATION
# *********************************************************************************************************************#


def augmented_mcc_evaluation():
    global mcc, matthews_set

    print("THIS IS MCC:", mcc)

    from augmented_load_data import All_text
    from augmented_training import tokenizer, model

    device = torch.device("cuda")

    #responses = All_text.RESPONSE.values
    #scores = All_text.CELF_SCORING.values
    responses = All_text.text.values
    scores = All_text.label.values

    print()
    print("Augmented - MCC Evaluation in mcc_evaluation.py")

    # Tokenize all sentences and map the tokens to their word IDs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for response in responses:
        # Encode_plus will...
        # 1) Tokenize the sentence
        # 2) Prepend the '[CLS]' token to the start.
        # 3) Append the '[SEP]' token to the send.
        # 4) Map the tokens ti their IDs.
        # 5) Pad or truncate the sentence to 'max_length'
        # 6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            response,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=44,  # Pad & truncate all sentences.--> FOR GREEK USE >44
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # Attention_masks differentiate relevant information from padding
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    scores = torch.tensor(scores)

    # Set the batch size
    batch_size = 256

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, scores)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Predicitons on the Test Set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_scores = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_scores = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        score_ids = b_scores.to('cpu').numpy()

        # remove nan from score_ids
        # score_ids = score_ids[~np.isnan(score_ids)]

        # Store predictions and true labels
        predictions.append(logits)
        true_scores.append(score_ids)

    print('    DONE.')

    """
    print('Positive samples: %d of %d (%.2f%%)' % (All_text.CELF_SCORING.sum(),
                                                   len(All_text.CELF_SCORING),
                                                   (All_text.CELF_SCORING.sum() /
                                                    len(All_text.CELF_SCORING) * 100.0)))"""

    print('Positive samples: %d of %d (%.2f%%)' % (All_text.label.sum(),
                                                  len(All_text.label),
                                                  (All_text.label.sum() /
                                                  len(All_text.label)* 100.0)))


    #******************************************************************************************************************#
    # Augmented # THE MCC-SET
    #******************************************************************************************************************#


    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating MCC for each batch...')

    # For each batch...
    for i in range(len(true_scores)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_scores_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        # if true_scores[i] != "nan"

        matthews = matthews_corrcoef(true_scores[i], pred_scores_i)
        matthews_set.append(matthews)

    print('DONE')


    #******************************************************************************************************************#
    # AUGMENTED # Final MCC Score
    #******************************************************************************************************************#

    # Combine all results across all batches
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score,
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_scores = np.concatenate(true_scores, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_scores, flat_predictions)

    print('Total augmented MCC: %.3f' % mcc)

    print("the mcc is : ", mcc)


print(mcc)