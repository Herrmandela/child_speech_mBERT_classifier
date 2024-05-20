import pandas as pd

print()
print()
print("|| +++ load_data.py +++ loaded ||")

#**********************************************************************************************************************#
# Define responses, scores, dataframes
#**********************************************************************************************************************#


responses = []
 
scores = []

df = []

#**********************************************************************************************************************#
# Load - English- *****
#**********************************************************************************************************************#

def load_english():
    global responses, scores, df
# Load the dataset into a pandas dataframe.
    print()
    print("Loading English ++++++ in load_data.py")
    print()

    df = pd.read_csv("/englishData.csv",
                     on_bad_lines='skip',
                     encoding='ISO-8859-1'
                     )
    

    df.loc[df.TOLD_SCORING ==0].sample(5)[['STRUCTURE','TARGET','RESPONSE','CELF_SCORING','TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values


    return responses, scores, df

#**********************************************************************************************************************#
# Load - FARSI - *****
#**********************************************************************************************************************#

def load_farsi():

    global responses, scores, df

    print()
    print("Loading Farsi ++++++ in load_data.py")
    print()

    df = pd.read_excel("/farsiMLC.xls")


    df = df.drop(['ChildID','STRUCTURE', 'Test Number',
                  'RESPONSEACT', 'Auto Scoring', 'TOLD_SCORING',
                  'MLC_CELF', 'MLC_TOLD'], axis = 1)

#    df = df.rename(columns={"TOLD-scoring": "TOLD_SCORING", "Scored Response": "RESPONSE", "Target": "TARGET"})

    df.dropna()

    responses = df.RESPONSE.values
    scores = df.CELF_SCORING.values

    print(df)

    return responses, scores, df


#**********************************************************************************************************************#
# Load - GREEK - *****
#**********************************************************************************************************************#

def load_greek():

    global responses, scores, df

    print()
    print("Loading Greek ++++++ in load_data.py")
    print()

    df = pd.read_csv("/greekData.csv",
                     on_bad_lines='skip')

#    df.loc[df.TOLD_SCORING == 0].sample(5)[['STRUCTURE', 'RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores, df


#**********************************************************************************************************************#
# Load - Multilingual - *****
#**********************************************************************************************************************#


def load_all():

    global responses, scores, df

    print()
    print("Loading Melange ++++++ in load_data.py")
    print()

    df = pd.read_csv("/allData.csv",
                     keep_default_na=False,
                     sep = ";")

    #df.loc[df.TOLD_SCORING == 0].sample(5)[['RESPONSE', 'CELF_SCORING', 'TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

    return responses, scores, df
