# import pandas as pd
print()
print()
print("|| +++ augmented_load_data.py +++ loaded ||")
print()
print()
#**********************************************************************************************************************#
# Defining the PATH, tokenizer, responses and scores.
#**********************************************************************************************************************#

responses = " "
scores = " "

PATH = "/Users/ph4533/Desktop/PyN4N/gitN4N/mBERT"
tokenizer = AutoTokenizer.from_pretrained(PATH)


#**********************************************************************************************************************#
# Load - English- *****
#**********************************************************************************************************************#

def augmented_load_english():
# Load the dataset into a pandas dataframe.  # 1

    print()
    print("1")
    print("Loading English ++++++ in augmented_load_data.py")

    df = pd.read_csv("/Users/ph4533/Desktop/PyN4N/gitN4N/datasets/ErrorTypesANON/New_3_SRep_all_data_withErrorTypesANON.csv",
                     on_bad_lines='skip',
                     encoding='ISO-8859-1')

#    df.loc[df.TOLD_SCORING ==0].sample(5)[['STRUCTURE','TARGET','RESPONSE','CELF_SCORING','TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores


#**********************************************************************************************************************#
# Load - FARSI - *****
#**********************************************************************************************************************#


def augmented_load_farsi():           # 1

    print()
    print("1")
    print("Loading Farsi ++++++ in augmented_load_data.py")


    df = pd.read_csv("/Users/ph4533/Desktop/PyN4N/gitN4N/datasets/Farsi/FarsiMLC.xls",
                     on_bad_lines='skip')

    df = df.drop(['ChildID', 'Sentence Type ', 'Test Number', 'Actual Response',
         'Notes', 'Auto Scoring', 'Syntactic Structure Score', 'Lexical Errors',
         'Word order errors ','Comments '],axis=1)

#    df = df.rename(columns={"TOLD-scoring": "TOLD_SCORING", "Scored Response": "RESPONSE", "Target": "TARGET"})

    df.dropna()

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores


#**********************************************************************************************************************#
# Load - GREEK - *****
#**********************************************************************************************************************#


def augmented_load_greek():         # 1

    print()
    print("1")
    print("Loading Greek ++++++ in augmented_load_data.py")
    print()

    df = pd.read_csv("/Users/ph4533/Desktop/PyN4N/gitN4N/datasets/Greek/SRT_GreekANON.csv",
                     on_bad_lines='skip')

#    df.loc[df.TOLD_SCORING == 0].sample(5)[['STRUCTURE', 'RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores

#**********************************************************************************************************************#
# Load - Multilingual - *****
#**********************************************************************************************************************#

def augmented_load_all():           # 1

    print()
    print("1")
    print("Loading Melange ++++++ in augmented_load_data.py")
    print()


    df = pd.read_csv("/Users/ph4533/Desktop/PyN4N/gitN4N/datasets/ErrorTypesANON/AllDataCELF.csv",
                     keep_default_na=False,
                     sep = ";")

    #df.loc[df.TOLD_SCORING == 0].sample(5)[['RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores


