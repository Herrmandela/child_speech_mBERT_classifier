import pandas as pd
print()
print()
print("|| +++ multiclass_load_data.py +++ loaded ||")

#**********************************************************************************************************************#
# Define responses, scores, dataframes
#**********************************************************************************************************************#

responses = []

scores = []

df = []

ALL_LABELS = []

STRUCTURE_INDICES = []
LABEL_REPOSITORY = []
CELF_SCORING_INDICES = []
TOLD_SCORING_INDICES = []

#**********************************************************************************************************************#
# Load - English- *****
#**********************************************************************************************************************#

def multiclass_load_english():      # 1

    global responses, scores, df, ALL_LABELS, STRUCTURE_INDICES, LABEL_REPOSITORY, CELF_SCORING_INDICES, TOLD_SCORING_INDICES

    print("1")
    print("Loading English ++++++ in multiclass_load_data.py")

    df = pd.read_csv("/english/englishData.csv",
                     on_bad_lines='skip',
                     encoding='ISO-8859-1'
                     )

    #df.loc[df.TOLD_SCORING ==0].sample(5)[['STRUCTURE','TARGET','RESPONSE','CELF_SCORING','TOLD_SCORING']]

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['CELF_SCORING'] = df['CELF_SCORING'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    responses = df.RESPONSE.values
    scores = df.CELF_SCORING.values

    LABEL_REPOSITORY = {
        "S_SVO+1_Aux": "SVO with ONE auxiliary/modal",
        "S_WH-quest.": "who, what, which object questions",
        "S_Long_Pass.": "Long actional Passive",
        "S_Adjunct": "Sentential Adjuncts",
        "S_Obj.Rel_RB": "Right Branching Object Relative clauses",
        "S_SVO+2_Aux": "SVO with TWO auxiliary/modal",
        "S_Short_Pass.": "Short actional passives",
        "S_Cond.": "Conditionals",
        "S_Obj.Rel_CE": "Centre Embedding Object Relative clauses",
        "C_0": "0",
        "C_1": "1",
        "C_2": "2",
        "C_3": "3",
        "T_0": "Incorrect",
        "T_1": "Correct",
    }

    STRUCTURE_LABELS = ['S_SVO+1_Aux', 'S_WH-quest.', 'S_Long_Pass.', 'S_Adjunct',
                        'S_Obj.Rel_RB', 'S_SVO+2_Aux', 'S_Short_Pass.', 'S_Cond.',
                        'S_Obj.Rel_CE']

    STRUCTURE_INDICES = range(0, 9)
    CELF_SCORING_LABELS = ["C_0","C_1","C_2","C_3"]
    CELF_SCORING_INDICES = range(9, 13)
    TOLD_SCORING_LABELS = ["T_0","T_1"]
    TOLD_SCORING_INDICES = range(13, 15)

    ALL_LABELS = STRUCTURE_LABELS + CELF_SCORING_LABELS + TOLD_SCORING_LABELS


    return df, ALL_LABELS, STRUCTURE_INDICES,LABEL_REPOSITORY, CELF_SCORING_INDICES,TOLD_SCORING_INDICES, responses, scores


#**********************************************************************************************************************#
# Load - FARSI - *****
#**********************************************************************************************************************#

def multiclass_load_farsi():           # 1

    global responses, scores, df, ALL_LABELS, STRUCTURE_INDICES, LABEL_REPOSITORY, CELF_SCORING_INDICES, TOLD_SCORING_INDICES

    print("1")
    print("Loading Farsi ++++++ in multiclass_load_data.py")

    df = pd.read_excel("/farsi/farsiMLC.xls",
                     on_bad_lines='skip')

    df.dropna()

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['CELF_SCORING'] = df['CELF_SCORING'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    responses = df.RESPONSE.values
    scores = df.CELF_SCORING.values

    LABEL_REPOSITORY = {
        "S_Posessive_Clitic": "Posessive_Clitic",
        "S_WH-quest." : "WH-quest.",
        "S_Obj.Rel_RB" : "Obj.Rel_RB",
        "S_Obj.Rel_CE" : "Obj.Rel_CE",
        "S_Complex_Ezafe" : "Complex_Ezafe",
        "S_Cond.":"Cond.",
        "S_Adjunct":"Adjunct",
        "S_Present_Progressive":"Present_Progressive",
        "S_Objective_Clitic":"Objective_Clitic",
        "C_0": "0",
        "C_1": "1",
        "C_2": "2",
        "C_3": "3",
        "T_0": "Incorrect",
        "T_1": "Correct",
    }

    STRUCTURE_LABELS = ["S_Posessive_Clitic", "S_WH-quest.", "S_Obj.Rel_RB", "S_Obj.Rel_CE",
                        "S_Complex_Ezafe", "S_Cond.", "S_Adjunct", "S_Present_Progressive",
                        "S_Objective_Clitic"]

    STRUCTURE_INDICES = range(0, 9)
    CELF_SCORING_LABELS = ["C_0", "C_1", "C_2", "C_3"]
    CELF_SCORING_INDICES = range(8, 13)
    TOLD_SCORING_LABELS = ["T_0", "T_1"]
    TOLD_SCORING_INDICES = range(13, 15)

    ALL_LABELS = STRUCTURE_LABELS + CELF_SCORING_LABELS + TOLD_SCORING_LABELS

    return df, ALL_LABELS, STRUCTURE_INDICES,LABEL_REPOSITORY, CELF_SCORING_INDICES,TOLD_SCORING_INDICES, responses, scores

#**********************************************************************************************************************#
# Load - GREEK - *****
#**********************************************************************************************************************#
def multiclass_load_greek():        # 1

    global responses, scores, df, ALL_LABELS, STRUCTURE_INDICES, LABEL_REPOSITORY, CELF_SCORING_INDICES, TOLD_SCORING_INDICES

    print("1")
    print("Loading Greek ++++++ in multiclass_load_data.py")

    df = pd.read_csv("/greek/greekData.csv",
                     on_bad_lines='skip')

    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    df['CELF_SCORING'] = df['CELF_SCORING'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    responses = df.RESPONSE.values
    scores = df.CELF_SCORING.values

    LABEL_REPOSITORY = {
        "SVO" : "SVO",
        "S_Negationn" : "Negation",
        "S_CLLD_CD": "CLLD_CD",
        "S_Coord." : "Coordination",
        "S_Comp_Clauses": "Complement Clauses",
        "S_Adverbials": "Adverbials",
        "S_WH-quest.": "who, what, which object questions",
        "S_Rel_Clauses": "Relative clauses",
        "C_0": "0",
        "C_1": "1",
        "C_2": "2",
        "C_3": "3",
        "T_0": "Incorrect",
        "T_1": "Correct",
    }

    STRUCTURE_LABELS = ["SVO", "S_Negationn", "S_CLLD_CD", "S_Coord.", "S_Comp_Clauses",
                        "S_Adverbials", "S_WH-quest.", "S_Rel_Clauses"]

    STRUCTURE_INDICES = range(0, 8)
    CELF_SCORING_LABELS = ["C_0", "C_1", "C_2", "C_3"]
    CELF_SCORING_INDICES = range(8, 12)
    TOLD_SCORING_LABELS = ["T_0", "T_1"]
    TOLD_SCORING_INDICES = range(12, 14)

    ALL_LABELS = STRUCTURE_LABELS + CELF_SCORING_LABELS + TOLD_SCORING_LABELS

    return df, ALL_LABELS, STRUCTURE_INDICES,LABEL_REPOSITORY, CELF_SCORING_INDICES,TOLD_SCORING_INDICES, responses, scores

#**********************************************************************************************************************#
# Load - Multilingual - *****
#**********************************************************************************************************************#
def multiclass_load_all():      # 1

    global responses, scores, df, ALL_LABELS, STRUCTURE_INDICES, LABEL_REPOSITORY, CELF_SCORING_INDICES, TOLD_SCORING_INDICES

    print()
    print("1")
    print("Loading Melange ++++++ in multiclass_load_data.py")

    df = pd.read_csv("/multilingual/allData.csv",
                     keep_default_na=False,
                     sep = ";")

    #df.loc[df.TOLD_SCORING == 0].sample(5)[['RESPONSE', 'TOLD_SCORING']]

    responses = df.RESPONSE.values
    scores = df.TOLD_SCORING.values

# **********************************************************************************************************************#
# Defining and Loading Multilingual Labels - *****
# **********************************************************************************************************************#

    print()
    print("2")
    print("Melange ** Define Label Repository ++++++ in multiclass_load_data.py")
    print()

    LABEL_REPOSITORY = {
        "T_0": "Incorrect",
        "T_1": "Correct",
    }

    TOLD_SCORING_LABELS = ["T_0", "T_1"]
    TOLD_SCORING_INDICES = range(0, 2)

    ALL_LABELS = TOLD_SCORING_LABELS

    return df, ALL_LABELS, STRUCTURE_INDICES,LABEL_REPOSITORY, CELF_SCORING_INDICES,TOLD_SCORING_INDICES, responses, scores

print("|| multiclass Load English ||")