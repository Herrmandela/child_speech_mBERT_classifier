# import pandas as pd

print()
print()
print("|| +++ load_data.py +++ loaded ||")
# print()
# print()

responses = " "
scores = " "

#**********************************************************************************************************************#
# Load - English- *****
#**********************************************************************************************************************#

def load_english():
# Load the dataset into a pandas dataframe.
    print()
    print("Loading English ++++++ in load_data.py")
    print()

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

def load_farsi():

    print()
    print("Loading Farsi ++++++ in load_data.py")
    print()

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

def load_greek():

    print()
    print("Loading Greek ++++++ in load_data.py")
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


def load_all():

    print()
    print("Loading Melange ++++++ in load_data.py")
    print()

    df = pd.read_csv("/Users/ph4533/Desktop/PyN4N/gitN4N/datasets/ErrorTypesANON/AllDataCELF.csv",
                     keep_default_na=False,
                     sep = ";")

    #df.loc[df.TOLD_SCORING == 0].sample(5)[['RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores
