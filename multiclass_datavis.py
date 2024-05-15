import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#**********************************************************************************************************************#
# Build MCC
#**********************************************************************************************************************#
# 17 in data_visualization.py
# 21 for Multiclass Models


def multiclass_plotMCC(user_input, depth_choice, experiment_choice):

    from mcc_evaluation import matthews_set, mcc
    print()
    print("Binary or Multiclass - MCC Plot +++++ in data_visualization.py")

    # Creat a barplot of the batches' MCC scores
    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, hue=matthews_set,
                     palette="blend:#7AB,#EDA", legend=False)

    ax.hlines(mcc, *ax.get_xlim())
    ax.annotate(f'Total MCC:\n {mcc:.3f}', xy=(ax.get_xlim()[1], mcc))

    plt.title(f"{experiment_choice} {depth_choice} {user_input}  MCC Score per Batch")
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')
    plt.show()

    # print(experiment_choice, depth_choice, user_input, 'MCC Score per Batch')
#**********************************************************************************************************************#
#  Multiclass Confusion Matrix
#**********************************************************************************************************************#

def multiclass_confusion_matrix(experiment_choice, depth_choice, user_input):       # 17

    from multiclass_training import y_preds
    from multiclass_data_prep import texten

    print()
    print("17")
    print(" MultiClass confusion Matrix +++++ in data_visualization.py")

    # With the predictions, we can plot the confusion matrix.
    cm = confusion_matrix(texten['val']['label'], y_preds, normalize="true")

    labels = ['incorrect', 'pass', 'acceptable ', 'correct']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False)

    plt.grid(False)

    plt.title(f"{experiment_choice} {depth_choice} {user_input} Model - Confusion Matrix")

    plt.show()

#**********************************************************************************************************************#
# Multi-classification training and validation polt.
#**********************************************************************************************************************#

def multiclass_training_and_validation_plot(experiment_choice, depth_choice, user_input):        # 18

    from multiclass_training import Macro_f1_CELF, Macro_f1_SENT, Macro_f1_TOLD, eval_loss, train_loss

    print("18")
    print("multiclass training and validation plot +++++ in data_visualization.py")
    print()

    import matplotlib.pyplot as plt
    #%matplotlib inline

    import seaborn as sns

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(eval_loss, 'g-*', label= 'Validation')
    plt.plot(train_loss, 'r-o', label= 'Training')

    plt.plot(Macro_f1_SENT, 'b-+', label='Sentence_structure_f1')
    plt.plot(Macro_f1_TOLD,'m-x', label='TOLD_f1')
    plt.plot(Macro_f1_CELF,'k-_', label='CELF_f1')


    # Label the plot
    plt.title(f"{experiment_choice} {depth_choice} {user_input} Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.xticks()

    plt.show()
