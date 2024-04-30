#**********************************************************************************************************************#
# # Define Training Arguments for the Augmented-Models
#**********************************************************************************************************************#
print()
print("12")
print("Compute Metrics +++ In augmented_metrics.py")
# Define a function to compute two metrics--accuracy and f1 score
def compute_metrics(pred):
  # True labels
  labels = pred.label_ids

  preds = pred.predictions.argmax(-1)
  # Note: average = "weighted" will weigh the f1_score by class sample size
  f1 = f1_score(labels, preds, average = "weighted")
  acc = accuracy_score(labels, preds)
  # Note: Need to return a dictionary
  return {"accuracy": acc, "f1": f1}

def preds_output(trainer):      # 12 for Augmented

  preds_output = trainer.predict(text_encoded["val"])

  preds_output.metrics

  y_preds = np.argmax(preds_output.predictions, axis = 1)

  # Model prediction metrics on the test set
  preds_output = trainer.predict(text_encoded["test"])
  preds_output.metrics