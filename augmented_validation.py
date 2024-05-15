All_text = []

def augmented_validation():

    global All_text
    print()
    print("4")
    print("creating Special Validation Set ++++++ in augmented_validation.py")

    All_text = pd.read_csv("/Users/ph4533/Desktop/PyN4N/Py38/gn4n/data/multilingual/allDataCELF.csv",
                     keep_default_na=False,
                     sep = ";")

    #All_text.loc[All_text.CELF_SCORING ==0].sample(5)[['TARGET','RESPONSE','CELF_SCORING','TOLD_SCORING']]

    print('Number of test sentences: {:,}\n'.format(All_text.shape[0]))

    All_text = All_text.rename(columns = {"CELF_SCORING": "label", "RESPONSE": "text"})
    All_text = All_text.drop(['AUTO_SCORING','TOLD_SCORING'], axis=1)
    All_text = All_text.dropna()
    All_text['label'] = All_text['label'].replace([3, 1, 2, 0],[0, 1, 2, 3])

    return All_text