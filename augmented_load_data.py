import pandas as pd
from transformers import BertTokenizer

print()
print()
print("|| +++ augmented_load_data.py +++ loaded ||")

#**********************************************************************************************************************#
# Defining the PATH, tokenizer, responses and scores.
#**********************************************************************************************************************#

df = []

All_text = []

#**********************************************************************************************************************#
# Load - English- *****
#**********************************************************************************************************************#

def augmented_load_english():
# Load the dataset into a pandas dataframe.  # 1

    global df
    print()
    print("1")
    print("Loading English ++++++ in augmented_load_data.py")

    df = pd.read_csv("/englishData.csv",
                     on_bad_lines='skip',
                     encoding='ISO-8859-1')

    df = df.rename(columns = {"CELF_SCORING": "label", "RESPONSE": "text"})

    df = df.drop(['ORDER', 'STRUCTURE', 'CODE', 'ITEM', 'Error_Types',
                  'AUTO_SCORING', 'TOLD_SCORING', 'GRAMMATICAL',
                  'STRUCTURE_SCORE', 'GENDER', 'AGE', 'AGE_of_ONSET_EN',
                  'AGE_of_ONSET_other', 'AoO_bi', 'ENGL_STATUS', 'OTHER_LANG',
                  'TOTAL_TOLD', 'TOTDAL_CELF', 'TOTAL_STRUCTURE', 'TOTAL_CELF',
                  'SUBGROUP', 'MLC_CELF', 'MLC_TOLD'], axis = 1)

#    df.loc[df.TOLD_SCORING ==0].sample(5)[['STRUCTURE','TARGET','RESPONSE','CELF_SCORING','TOLD_SCORING']]

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['label'] = df['label'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    return df


#**********************************************************************************************************************#
# Load - FARSI - *****
#**********************************************************************************************************************#


def augmented_load_farsi():           # 1

    global df

    print()
    print("1")
    print("Loading Farsi ++++++ in augmented_load_data.py")


    df = pd.read_excel("/farsiMLC.xls")


    df = df.rename(columns = {"CELF_SCORING": "label", "RESPONSE": "text"})

#    df = df.rename(columns={"TOLD-scoring": "TOLD_SCORING", "Scored Response": "RESPONSE", "Target": "TARGET"})

    df = df.drop(['ChildID','STRUCTURE', 'Test Number', 'RESPONSEACT',
                'Auto Scoring', 'MLC_CELF', 'MLC_TOLD'], axis = 1)
    df.dropna()

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['label'] = df['label'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    return df


#**********************************************************************************************************************#
# Load - GREEK - *****
#**********************************************************************************************************************#


def augmented_load_greek():         # 1

    global df

    print()
    print("1")
    print("Loading Greek ++++++ in augmented_load_data.py")
    print()

    df = pd.read_csv("/greekData.csv",
                     on_bad_lines='skip',
                     sep = ",")

    df = df.rename(columns = {"CELF_SCORING": "label", "RESPONSE": "text"})

    df = df.drop(['ChildID', 'STRUCTURE' , 'TEST_NUMBER' ,'AUTO_SCORING',
                  'MLC_CELF','MLC_TOLD'], axis = 1)

#    df.loc[df.TOLD_SCORING == 0].sample(5)[['STRUCTURE', 'RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['label'] = df['label'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    return df

#**********************************************************************************************************************#
# Load - Multilingual - *****
#**********************************************************************************************************************#

def augmented_load_all():           # 1

    global responses, scores, df

    print()
    print("1")
    print("Loading Melange ++++++ in augmented_load_data.py")
    print()


    df = pd.read_csv("/allDataCELF.csv",
                     keep_default_na=False,
                     sep = ";")

    df = df.rename(columns = {"CELF_SCORING": "label", "RESPONSE": "text"})

    #df.loc[df.TOLD_SCORING == 0].sample(5)[['RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]$

    df = df.drop(['AUTO_SCORING', 'TOLD_SCORING'], axis = 1)
    df = df.dropna()

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['label'] = df['label'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    return df

#**********************************************************************************************************************#
# Load -Validation Data - *****
#**********************************************************************************************************************#


def augmented_load_validation():

    global All_text

    print("creating Special Validation Set ++++++ in augmented_validation.py")

    All_text = pd.read_csv("/allDataCELF.csv",
                          keep_default_na=False,
                          sep = ";")

    All_text = All_text.rename(columns={"CELF_SCORING": "label", "RESPONSE": "text"})


    All_text = All_text.drop(['AUTO_SCORING', 'TOLD_SCORING'], axis=1)
    All_text = All_text.dropna()

    print('Number of test sentences: {:,}\n'.format(All_text.shape[0]))

    All_text['label'] = All_text['label'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    return All_text

augmented_load_validation()