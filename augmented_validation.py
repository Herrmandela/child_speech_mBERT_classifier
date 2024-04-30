def augmented_validation():

    print()
    print("4")
    print("creating Special Validation Set ++++++ in augmented_validation.py")

    all_test = pd.read_csv("/content/drive/MyDrive/CNotebooks/ErrorTypesANON/AllDataCELF.csv",
                          keep_default_na=False,
                          sep=";")

    All_text = all_test.rename(columns={"CELF_SCORING": "label", "RESPONSE": "text"})

    All_text = All_text.drop(['AUTO_SCORING', 'TOLD_SCORING'], axis=1)

    All_text = All_text.dropna()

    return All_text