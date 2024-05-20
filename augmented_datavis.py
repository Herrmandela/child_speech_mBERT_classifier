import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#**********************************************************************************************************************#
# Augmented MccPlot
#**********************************************************************************************************************#

def augmented_mccPlot(experiment_choice, depth_choice, user_input):       # 18

    from mcc_evaluation import mcc, matthews_set

    # Creat a barplot of the batches' MCC scores
    sns.set(color_codes=True)

    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, hue=matthews_set,
                     palette="blend:#7AB,#EDA", legend=False)

    ax.hlines(mcc, *ax.get_xlim())
    ax.annotate(f'Total MCC:\n {mcc:.3f}', xy=(ax.get_xlim()[1], mcc))


    plt.title(f"{experiment_choice} {depth_choice} {user_input} Model - MCC Score per Batch")
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')
    plt.show()


#**********************************************************************************************************************#
# Augmented training and validation polt.
#**********************************************************************************************************************#


def augmented_training_and_validation_plot(experiment_choice, depth_choice, user_input):        # 12

    from augmented_training import train_loss, Macro_f1, accuracy, val_loss

    import matplotlib.pyplot as plt
    #matplotlib inline

    import seaborn as sns
    sns.set_context("paper")
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(val_loss, 'g-*', label='Validation')
    plt.plot(train_loss, 'r-o', label='Training')

    plt.plot(Macro_f1, 'y-*', label='F1')
    plt.plot(accuracy, 'b-+', label='Accuracy')

    # Label the plot
    plt.title(f"{experiment_choice} {depth_choice} {user_input} Model - Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.xticks([1, 2, 3, 4, 5])

    plt.show()


#**********************************************************************************************************************#
# Augmented Confusion Matrix
#**********************************************************************************************************************#


def augmented_confusion_matrix(experiment_choice, depth_choice, user_input):        # 13

    from augmented_training import y_preds, text_encoded 

    # With the predictions, we can plot the confusion matrix.
    cm = confusion_matrix(text_encoded["val"]['label'], y_preds, normalize="true")

    labels = ['incorrect', 'pass', 'acceptable ', 'correct']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="PuBu", values_format=".2f", colorbar=False)
    # disp.plot(cmap = "Set2", values_format = ".2f", colorbar = False)
    # disp.plot(cmap = "tab10", values_format = ".2f", colorbar = False)

    plt.grid(False)

    plt.title(f"{experiment_choice} {depth_choice} {user_input} Model - Confusion Matrix")

    plt.show()
