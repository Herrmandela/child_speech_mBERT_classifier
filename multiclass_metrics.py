"""
During the training phase logits are built, from which the HugginFace transformer
will deduce relations and predicted vectors.
They will be compared with the labels and returned as framed metrics including accuracy,
precision, recall and f1-scores.
in multiclass classification the predicted vector is reached by assigning a 1 to
the highest logit and 0 to all others, using the softMax the class with the highest
probability will have the highest score.
In multilabel classification deduces the predicted vector by assigning 1 to all
non-negative logits and 0 to all negative logits. >=0.5 probability belongs
to the corresponding class.
"""


def get_preds_from_logits(logits):    # 8

    print()
    print("9")
    print("Getting Predictions from Logits")


    ret = np.zeros(logits.shape)

    # First 5 columns are handles with a multi-class approach - 1 class fills the highest probability.
    best_class = np.argmax(logits[:, STRUCTURE_INDICES], axis=-1)
    ret[list(range(len(ret))), best_class] = 1

    # The other columns are for the scores
    ret[:, CELF_SCORING_INDICES] = (logits[:, CELF_SCORING_INDICES] >= 0).astype(int)
    ret[:, TOLD_SCORING_INDICES] = (logits[:, TOLD_SCORING_INDICES] >= 0).astype(int)

    return ret


def compute_metrics(eval_pred):                 # 9

    print()
    print("10")
    print("Defining Metrics from Logits")

    logits, labels = eval_pred
    final_metrics = {}

    # Deduce predictions from logits
    predictions = get_preds_from_logits(logits)

    # Get f1 metric for STRUCTURE --> f1_micro == accuracy.
    final_metrics["f1_micro_for_sentence_structure"] = f1_score(labels[:, STRUCTURE_INDICES],
                                                                predictions[:, STRUCTURE_INDICES], average="micro")
    final_metrics["f1_macro_for_sentence_structure"] = f1_score(labels[:, STRUCTURE_INDICES],
                                                                predictions[:, STRUCTURE_INDICES], average="macro")

    # Get f1 metric for CELF --> f1_micro == accuracy.
    final_metrics["f1_micro_for_CELF"] = f1_score(labels[:, CELF_SCORING_INDICES], predictions[:, CELF_SCORING_INDICES],
                                                  average="micro")
    final_metrics["f1_macro_for_CELF"] = f1_score(labels[:, CELF_SCORING_INDICES], predictions[:, CELF_SCORING_INDICES],
                                                  average="macro")

    # Get f1 metric for TOLD --> f1_micro == accuracy.
    final_metrics["f1_micro_for_TOLD"] = f1_score(labels[:, TOLD_SCORING_INDICES], predictions[:, TOLD_SCORING_INDICES],
                                                  average="micro")
    final_metrics["f1_macro_for_TOLD"] = f1_score(labels[:, TOLD_SCORING_INDICES], predictions[:, TOLD_SCORING_INDICES],
                                                  average="macro")

    # The global f1_metrics
    final_metrics["f1_micro"] = f1_score(labels, predictions, average="micro")
    final_metrics["f1_macro"] = f1_score(labels, predictions, average="macro")

    # Classification report
    print("Classification report for Structures: ")
    print(classification_report(labels[:, STRUCTURE_INDICES], predictions[:, STRUCTURE_INDICES], zero_division=0))
    print("Classification report for CELF_SCORES: ")
    print(classification_report(labels[:, CELF_SCORING_INDICES], predictions[:, CELF_SCORING_INDICES], zero_division=0))
    print("Classification report for TOLD_SCORES: ")
    print(classification_report(labels[:, TOLD_SCORING_INDICES], predictions[:, TOLD_SCORING_INDICES], zero_division=0))

    return final_metrics