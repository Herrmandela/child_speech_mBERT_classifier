from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainerCallback, TrainingArguments
#
import torch
from multiclass_data_prep import ds
from multiclass_metrics import *
from multiclass_load_data import ALL_LABELS, STRUCTURE_INDICES, CELF_SCORING_INDICES, TOLD_SCORING_INDICES

# print("ds: ",type(ds))
# print(ds)

y_preds = ""

model = ""

output_dir = ""

id2label = {k: l for k, l in enumerate(ALL_LABELS)}
label2id = {l: k for k, l in enumerate(ALL_LABELS)}

LEARNING_RATE = 2e-4
MAX_LENGTH = 44
BATCH_SIZE = 32

device = torch.device("cuda")

PATH = "/content/drive/MyDrive/data/mBERT"

#PATH = "/Users/ph4533/Desktop/PyN4N/Py38/gn4n/mBERT"

tokenizer = BertTokenizer.from_pretrained(PATH, do_lower_case=True)

model = BertForSequenceClassification.from_pretrained(
        PATH,  # Use the 12-layer BERT model, with an uncased vocab.
        id2label=id2label,  # A dictionary linking label to id
        label2id=label2id,  # A dictionary linking id to label
        )

num_train_epochs = input("Please choose number of Epochs: (We recommend 3-4)")
#**********************************************************************************************************************#
# # MODEL TRAINING DEPTH -  for layers training protocol
#**********************************************************************************************************************#

def multiclass_training_vanilla():          # 3

    global tokenizer, PATH, model, device

    model.cuda() # Tell the model to run on GPU

    print("model: ",type(model))
    print(model)


    return model


def multiclass_training_inherent():         # 3

    print()
    print("3")
    print("Define Model Parameters")
    print('The MultiClassification Inherent Model was Chosen ++++ in multiclass_training.py')
    print()

    global tokenizer, PATH, model, device
    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    ## inherent - Only freezes the input and output layers (pooler)

    list(model.parameters())[5].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                        # Tell the model to run on GPU

    return model


def multiclass_training_shallow():          # 3

    print()
    print("3")
    print("Define Model Parameters")
    print('The MultiClassification Shallow Model was Chosen ++++ in multiclass_training.py')


    global tokenizer, PATH, model, device

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    # ShallowFreeze+pooler

    list(model.parameters())[53].requires_grad = True
    list(model.parameters())[-4].requires_grad = True

    model.cuda()                        # Tell the model to run on GPU

    return model



#**********************************************************************************************************************#
# # MODEL TRAINING
#**********************************************************************************************************************#

class MultiTaskClassificationTrainer(Trainer):  # 10 in multiclass_training.py

    global model

    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights

    def compute_loss(self, model, inputs, return_outputs=False):  # 11 in multiclass_training.py


        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]

        structure_loss = torch.nn.functional.cross_entropy(logits[:, STRUCTURE_INDICES],
                                                            labels[:, STRUCTURE_INDICES].float())
        TOLD_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, TOLD_SCORING_INDICES],
                                                                          labels[:, TOLD_SCORING_INDICES].float())
        CELF_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, CELF_SCORING_INDICES],
                                                                          labels[:, CELF_SCORING_INDICES].float())

        loss = self.group_weights[0] * structure_loss + self.group_weights[2] * TOLD_loss + self.group_weights[
            1] * CELF_loss
        return (loss, outputs) if return_outputs else loss



class PrinterCallback(TrainerCallback):  # 12 in multiclass_training.py
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):  # 13 multiclass_training.py
        print(f"Epoch {state.epoch}: ")


output_dir = input("Please enter the output directory: ")


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=float(num_train_epochs),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="f1_macro",
    load_best_model_at_end=True,
    weight_decay=0.01
    )


trainer = MultiTaskClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
    callbacks=[PrinterCallback],
    group_weights=(0.7, 4, 4)
    )

trainer.train()

trainer.evaluate()

Macro_f1_SENT = []
for elem in trainer.state.log_history:
    if 'eval_f1_macro_for_sentence_structure' in elem.keys():
        Macro_f1_SENT.append(elem['eval_f1_macro_for_sentence_structure'])

Macro_f1_CELF = []
for elem in trainer.state.log_history:
    if 'eval_f1_macro_for_CELF' in elem.keys():
        Macro_f1_CELF.append(elem['eval_f1_macro_for_CELF'])

Macro_f1_TOLD = []
for elem in trainer.state.log_history:
    if 'eval_f1_macro_for_TOLD' in elem.keys():
        Macro_f1_TOLD.append(elem['eval_f1_macro_for_TOLD'])


eval_loss = []
for elem in trainer.state.log_history:
    if 'eval_loss' in elem.keys():
        eval_loss.append(elem['eval_loss'])

train_loss = []
for elem in trainer.state.log_history:
    if 'loss' in elem.keys():
        train_loss.append(elem['loss'])

#**********************************************************************************************************************#
# # Preds_output
#**********************************************************************************************************************#

preds_output = trainer.predict(ds["validation"])

preds_output.metrics

y_preds = np.argmax(preds_output.predictions, axis=1)

preds_output = trainer.predict(ds["test"])

preds_output.metrics



