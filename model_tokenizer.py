from transformers import BertTokenizer
import numpy as np

print()
print("Binary +++ model_tokenizer.py")
print()



global responses, scores


# # Define PATH
#PATH = "/Users/ph4533/Desktop/PyN4N/gitN4N/mBERT"


#**********************************************************************************************************************#
# # Load the Tokenizer
#**********************************************************************************************************************#


print('Loading the mBERT Tokenizer')

tokenizer = BertTokenizer.from_pretrained(PATH, do_lower_case=True)


# # Formatting required:


responses = np.delete(responses, np.where(responses == 'nan'))

# # Tokenize the Dataset - using the encode function for parsing and data-prep needs.


max_len = 0

# # iterate through the sentences
for response in responses:

# Tokenize the text and add '[CLS]' and '[SEP]' tokens.
    input_ids = tokenizer.encode(response, add_special_tokens=True)

# Update the maximum sentence length - so that it is as long as the longest input ID
    max_len = max(max_len, len(input_ids))

# print the Max_length to adjust model parameters.
print('Max sentence length: ', max_len)





# # Tokenize the responses - all sentences and map the output to word_IDs

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
      response,                           # Sentence to encode.
      add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
      max_length = 60,                # Pad & truncate all sentences.
      pad_to_max_length = True,
      return_attention_mask = True,   # Construct attn. masks.
      return_tensors = 'pt',          # Return pytorch tensors.
    )

# Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

# Attention_masks differentiate relevant information from padding
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
scores = torch.tensor(scores)

# Print sentence 0, now as a list of IDs.
# print(' Original: ', responses[0])
# print(' Token Ids: ', input_ids[0])