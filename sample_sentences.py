input_texts = [
               "Who have they seen near the steps?",
               "He should wash the baby that the child is patting.",
               "What did they find yesterday in the snow?",
               "The books...",
               "The children will if they sleep.",
               "They are eating the bananas in the dark.",
               "οι οδηγοί άφησαν τους επιβάτες των λεωφορείων στην επόμενη στάση",
               "ο χορευτής πήρε την ομπρέλα του και περπάτησε στη δυνατή βροχή",
               "ο προπονητής δεν να κερδίσει η ομάδα του σήμερα",
               "η γιαγιά ότι σε αυτά τα μέρη πετούσαν περίεργα πουλιά",
               "dɔxtær-i kɛ tɔ dust dɑri xɑhærɛ mænɛ",
               "kudum kɛtɑb rɔ ɛntɛxɑb kærd?",
               "væqti lɛbæs pushidam mikhoram",
               "kahli",
               "Es war eine lange dunkle Nacht.",
               "È stata una lunga notte buia",
               "یہ ایک لمبی سیاہ رات تھی",
               "C'était une longue nuit sombre",
               "لقد كانت ليلة مظلمة طويلة",
               "أحب المدرسة"
               ]

# greek_text = ["οι εφημερίδες γράφουν πολλά για τον ληστή που έπιασε η αστυνομία",]

def sampleSentences():

    print()
    print("Sample Sentences in sample_sentences.py")



    for text in input_texts:
      # Encode the text
      input = tokenizer(text, truncation=True, padding="max_length",
                        max_length=44, return_tensors="pt").to("cuda")
      with torch.no_grad():
        # Call the model to predict under the format of logits of 15 classes
        logits = model(**input).logits.cpu().detach().numpy()

      predicted_class_id = logits.argmax().item()

      prediction = model.config.id2label[predicted_class_id]

      print('The sentence: "',text, '"  is:', prediction)


def multiclass_sample_sentences():          # 19

    print()
    print("19")
    print("multiclass Samples in sample_sentences.py")

    # Encode the text
    encoded = tokenizer(input_texts, truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt").to("cuda")

    # Call the model to predict under the format of logits of 15 classes
    logits = model(**encoded).logits.cpu().detach().numpy()

    # Decode the result
    preds = get_preds_from_logits(logits)
    decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in preds]

    for text, pred in zip(input_texts, decoded_preds):
        print(text)
        print("STRUCTURE:", [LABEL_REPOSITORY[l] for l in pred if l.startswith('S')])
        print("CELF_SCORING:", [LABEL_REPOSITORY[l] for l in pred if l.startswith('C')])
        print("TOLD_SCORING:", [LABEL_REPOSITORY[l] for l in pred if l.startswith('T')]) # Exclude "no cause" for simpler reading
        print("")


def augmented_sample_sentences():           # 16

    print()
    print("augmented Samples in sample_sentences.py")


    print(augmentation_choice, " augmented", language," ++ ", modeltype, "Model - Sample Sentence Classification")
    for text in input_texts:
        prediction = classifier(text)
        print('the sentence: "', text, '" is:', prediction)