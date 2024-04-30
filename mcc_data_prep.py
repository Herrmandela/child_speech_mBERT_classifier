# from torch.utils.data import TensorDataset

#**********************************************************************************************************************#
# Preparing data for the MCC evaluation -
#**********************************************************************************************************************#
# 15!
def mcc_data_prep():


    print()
    print("Binary or multiclass - MCC Data Preparation in mcc_data_prep.py")


    # Load the dataset into a Pandas dataframe. Originally, this is where the
    # Multilingual data was loaded for model evaluation.

    df_test = pd.read_csv("/Users/ph4533/Desktop/PyN4N/gitN4N/datasets/ErrorTypesANON/AllDataCELF.csv",
                          keep_default_na=False,
                          sep=";")
    # Report the number of sentences.
    #print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))

    #df_test.columns

    # Create sentence and label lists
    responses = df_test.RESPONSE.values
    scores = df_test.TOLD_SCORING.values   # in the MultiClass and Augmented models this is CELF

    # Tokenize all of the sentences and map the tokens to their word IDs
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
            max_length=44,  # Pad & truncate all sentences.
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
    prediction_dataloader = DataLoader(prediction_data,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size)

   return prediction_data, prediction_sampler, prediction_dataloader,

