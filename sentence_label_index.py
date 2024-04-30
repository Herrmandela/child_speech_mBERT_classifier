def english_sentence_label_index():

    print()
    print("English sentence and Label Indices in sentence_label_index.py")
    print()


    STRUCTURE_LABELS = ['S_SVO+1_Aux', 'S_WH-quest.', 'S_Long_Pass.', 'S_Adjunct', 'S_Obj.Rel_RB', 'S_SVO+2_Aux',
                        'S_Short_Pass.', 'S_Cond.', 'S_Obj.Rel_CE']
    STRUCTURE_INDICES = range(0, 9)
    CELF_SCORING_LABELS = ["C_0", "C_1", "C_2", "C_3"]
    CELF_SCORING_INDICES = range(9, 13)
    TOLD_SCORING_LABELS = ["T_0", "T_1"]
    TOLD_SCORING_INDICES = range(13, 15)

    ALL_LABELS = STRUCTURE_LABELS + CELF_SCORING_LABELS + TOLD_SCORING_LABELS

    return ALL_LABELS, STRUCTURE_LABELS


def farsi_sentence_label_index():

    print()
    print("Farsi sentence and Label Indices in sentence_label_index.py")
    print()

    STRUCTURE_LABELS = ["S_Posessive_Clitic", "S_WH-quest.", "S_Obj.Rel_RB", "S_Obj.Rel_CE",
                        "S_Complex_Ezafe", "S_Cond.", "S_Adjunct", "S_Present_Progressive", "S_Objective_Clitic"]
    STRUCTURE_INDICES = range(0, 9)
    CELF_SCORING_LABELS = ["C_0", "C_1", "C_2", "C_3"]
    CELF_SCORING_INDICES = range(8, 13)
    TOLD_SCORING_LABELS = ["T_0", "T_1"]
    TOLD_SCORING_INDICES = range(13, 15)

    ALL_LABELS = STRUCTURE_LABELS + CELF_SCORING_LABELS + TOLD_SCORING_LABELS

    return ALL_LABELS, STRUCTURE_LABELS


def greek_sentence_label_index():

    print()
    print("Greek sentence and Label Indices in sentence_label_index.py")
    print()

    STRUCTURE_LABELS = ["SVO", "S_Negationn", "S_CLLD_CD", "S_Coord.", "S_Comp_Clauses", "S_Adverbials", "S_WH-quest.",
                        "S_Rel_Clauses"]
    STRUCTURE_INDICES = range(0, 8)
    CELF_SCORING_LABELS = ["C_0", "C_1", "C_2", "C_3"]
    CELF_SCORING_INDICES = range(8, 12)
    TOLD_SCORING_LABELS = ["T_0", "T_1"]
    TOLD_SCORING_INDICES = range(12, 14)

    ALL_LABELS = STRUCTURE_LABELS + CELF_SCORING_LABELS + TOLD_SCORING_LABELS

    return ALL_LABELS, STRUCTURE_LABELS


# def multilingual_sentence_label_index():
#
#     print()
#     print("Multilingual sentence and Label Indices in sentence_label_index.py")
#     print()
#
#     return ALL_LABELS





