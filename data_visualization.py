import matplotlib.pyplot as plt
import seaborn as sns
from mcc_evaluation import matthews_set, mcc


print()
print()
print("+++++++In data_visualization.py+++++++")



#**********************************************************************************************************************#
# Build Histogram for Experiment 2
#**********************************************************************************************************************#


def plotHistogram(): # This is the same for all languages.

    print()
    print("12.1.Histogram")
    print("Histogram Plot ++++++ in data_visualization.py")
    print()

    # plt.hist(df['CELF_SCORING'], color='yellow', edgecolor='black',
    #          bins=int(10))
    #
    # # Seaborn Histogram
    # sns.histplot(df['TOLD_SCORING'], kde=False,
    #              bins=int(10), color='pink')
    #
    # # Add Labels
    # plt.title('Histogram of Scoring')
    # plt.xlabel('Scores (min)')
    # plt.ylabel('Sentences')



    # Build Histogram for Experiment 2
def plotSidebyside_english():

    print()
    print("12.2.SideBySide_English")
    print("English SideBySide ++++++ in data_visualization.py")
    print()

    # x1 = list(df[df['STRUCTURE'] == 'S_SVO+1_Aux']['CELF_SCORING'])
    # x2 = list(df[df['STRUCTURE'] == 'S_WH-quest.']['CELF_SCORING'])
    # x3 = list(df[df['STRUCTURE'] == 'S_Long_Pass.']['CELF_SCORING'])
    # x4 = list(df[df['STRUCTURE'] == 'S_Adjunct']['CELF_SCORING'])
    # x5 = list(df[df['STRUCTURE'] == 'S_Obj.Rel_RB']['CELF_SCORING'])
    # x6 = list(df[df['STRUCTURE'] == 'S_SVO+2_Aux']['CELF_SCORING'])
    # x7 = list(df[df['STRUCTURE'] == 'S_Short_Pass.']['CELF_SCORING'])
    # x8 = list(df[df['STRUCTURE'] == 'S_Cond.']['CELF_SCORING'])
    # x9 = list(df[df['STRUCTURE'] == 'S_Obj.Rel_CE']['CELF_SCORING'])
    #
    # # Assign colors for each sentence structure
    # colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00', '#F0E562', '#D25F00', '#003E99', '#E23F11']
    # names = ['S_SVO+1_Aux', 'S_WH-quest.', 'S_Long_Pass.', 'S_Adjunct',
    #          'S_Obj.Rel_RB', 'S_SVO+2_Aux', 'S_Short_Pass.', 'S_Cond.',
    #          'S_Obj.Rel_CE']
    #
    # plt.hist([x1, x2, x3, x4, x5, x6, x7, x8, x9], bins=int(180 / 15),
    #          color=colors, label=names)
    #
    # # Stacked Plot
    # # plt.hist([x1, x2, x3, x4, x5, x6, x7, x8], bins=int(180 / 15), stacked=True,
    # #          color=colors, label=names)
    #
    # # Plot formatting
    # plt.legend()
    # plt.xlabel('Scores (min)')
    # plt.ylabel('Sentences')
    # plt.title('Side-by-Side Histogram with Structures')
    #
    
    
def plotSidebyside_farsi():

    print()
    print("12.2.SideBySide_Farsi")
    print("Farsi SideBySide ++++++ in data_visualization.py")
    print()

    # x1 = list(df[df['STRUCTURE'] == 'Posessive_Clitic']['CELF_SCORING'])
    # x2 = list(df[df['STRUCTURE'] == 'WH-quest.']['CELF_SCORING'])
    # x3 = list(df[df['STRUCTURE'] == 'Obj.Rel_RB']['CELF_SCORING'])
    # x4 = list(df[df['STRUCTURE'] == 'Obj.Rel_CE']['CELF_SCORING'])
    # x5 = list(df[df['STRUCTURE'] == 'Complex_Ezafe']['CELF_SCORING'])
    # x6 = list(df[df['STRUCTURE'] == 'Cond.']['CELF_SCORING'])
    # x7 = list(df[df['STRUCTURE'] == 'Adjunct']['CELF_SCORING'])
    # x8 = list(df[df['STRUCTURE'] == 'Present_Progressive']['CELF_SCORING'])
    # x9 = list(df[df['STRUCTURE'] == 'Objective_Clitic']['CELF_SCORING'])
    #
    # # Assign colors for each sentence structure
    # colors = ['#E69F00', '#56B4E9', '#2F4F4F', '#009E73', '#FF6347', '#F0E562', '#FF0000', '#003E99', '#E44F00']
    # names = ['Posessive_Clitic', 'WH-quest.', 'Obj.Rel_RB', 'Obj.Rel_CE',
    #          'Complex_Ezafe', 'Cond.', 'Adjunct',
    #          'Present_Progressive', 'Objective_Clitic']
    #
    # plt.hist([x1, x2, x3, x4, x5, x6, x7, x8, x9], bins=int(180 / 15),
    #          color=colors, label=names)
    #
    # #Stacked Plot
    # # plt.hist([x1, x2, x3, x4, x5, x6, x7, x8, x9], bins=int(180 / 15), stacked=True,
    # #          color=colors, label=names)
    #
    # # Plot formatting
    # plt.legend()
    # plt.xlabel('Scores (min)')
    # plt.ylabel('Sentences')
    # plt.title('Side-by-Side Histogram_Farsi Sentence Structures')


def plotSidebyside_greek():

    print()
    print("12.2.SideBySide_Greek")
    print("Greek SideBySide ++++++ in data_visualization.py")
    print()

    # x1 = list(df[df['STRUCTURE'] == 'SVO']['CELF_SCORING'])
    # x2 = list(df[df['STRUCTURE'] == 'S_Negationn']['CELF_SCORING'])
    # x3 = list(df[df['STRUCTURE'] == 'S_CLLD_CD']['CELF_SCORING'])
    # x4 = list(df[df['STRUCTURE'] == 'S_Coord.']['CELF_SCORING'])
    # x5 = list(df[df['STRUCTURE'] == 'S_Comp_Clauses']['CELF_SCORING'])
    # x6 = list(df[df['STRUCTURE'] == 'S_Adverbials']['CELF_SCORING'])
    # x7 = list(df[df['STRUCTURE'] == 'S_WH-quest.']['CELF_SCORING'])
    # x8 = list(df[df['STRUCTURE'] == 'S_Rel_Clauses']['CELF_SCORING'])
    #
    # # Assign colors for each sentence structure
    # colors = ['#E69F00', '#56B4E9', '#2F4F4F', '#009E73', '#FF6347', '#F0E562', '#FF0000', '#003E99']
    # names = ['SVO', 'S_Negationn', 'S_CLLD_CD', 'S_Coord.', 'S_Comp_Clauses',
    #          'S_Adverbials', 'S_WH-quest.', 'S_Rel_Clauses']
    #
    # plt.hist([x1, x2, x3, x4, x5, x6, x7, x8], bins=int(180 / 15),
    #          color=colors, label=names)
    #
    # #Stacked Plot
    # # plt.hist([x1, x2, x3, x4, x5, x6, x7, x8], bins=int(180 / 15), stacked=True,
    # #          color=colors, label=names)
    #
    # # Plot formatting
    # plt.legend()
    # plt.xlabel('Scores (min)')
    # plt.ylabel('Sentences')
    # plt.title('Side-by-Side Histogram_Greek Sentence Structures')

    
#**********************************************************************************************************************#
# Plot and Validation after training
#**********************************************************************************************************************#

def plotValidationAndLoss(user_input, depth_choice, experiment_choice):

    from training import df_stats

    print()
    print("Plotting Validation and Loss +++++ in DataVisualization.py")


    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label='Training')
    plt.plot(df_stats['Valid. Loss'], 'g-o', label='Validation')

    # Label the plot
    plt.title(f"{experiment_choice}, {depth_choice}, {user_input}, Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5])

    plt.show()


    # print(experiment_choice, depth_choice, user_input, 'Training & Validation Loss')


#**********************************************************************************************************************#
# Build MCC
#**********************************************************************************************************************#
# 17 in data_visualization.py
# 21 for Multiclass Models

def plotMCC(user_input, depth_choice, experiment_choice):

    from mcc_evaluation import matthews_set, mcc

    print()
    print("Binary or Multiclass - MCC Plot +++++ in data_visualization.py")

    # Creat a barplot of the batches' MCC scores
    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

    ax.hlines(mcc, *ax.get_xlim())
    ax.annotate(f'Total MCC:\n {mcc:.3f}', xy=(ax.get_xlim()[1], mcc))

    plt.title(f"{experiment_choice}, {depth_choice}, {user_input}, MCC Score per Batch")
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')

    plt.show()

    # print(experiment_choice, depth_choice, user_input, 'MCC Score per Batch')

#**********************************************************************************************************************#
#  Multiclass Confusion Matrix
#**********************************************************************************************************************#

def multiclass_confusion_matrix(experiment_choice, depth_choice, user_input):       # 17

    print()
    print("17")
    print(" MultiClass confusion Matrix +++++ in data_visualization.py")


    # With the predictions, we can plot the confusion matrix.
    cm = confusion_matrix(text_encoded["val"]['label'], y_preds, normalize="true")

    labels = ['incorrect', 'pass', 'acceptable ', 'correct']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False)

    plt.grid(False)

    plt.title(experiment_choice, depth_choice, user_input," Confusion Matrix")


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


    print(experiment_choice, depth_choice, user_input, "Model")

    return Macro_f1_SENT, Macro_f1_TOLD, Macro_f1_CELF, eval_loss

#**********************************************************************************************************************#
# Multi-classification training and validation polt.
#**********************************************************************************************************************#

def multiclass_training_and_validation_plot(experiment_choice, depth_choice, user_input):        # 18

    print()
    print("18")
    print("multiclass training and validation plot +++++ in data_visualization.py")
    print()


    import matplotlib.pyplot as plt
    %matplotli inline

    import seaborn as sns

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(eval_loss, label='Validation')

    plt.plot(Macro_f1_SENT, label='Sentence_structure_f1')
    plt.plot(Macro_f1_TOLD, label='TOLD_f1')
    plt.plot(Macro_f1_CELF, label='CELF_f1')

    # Label the plot
    plt.title(experiment_choice, depth_choice, user_input, "Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.xticks()

    plt.show()

    print(experiment_choice, depth_choice, user_input, "Training & Validation Loss")


#**********************************************************************************************************************#
# Augmented MccPlot
#**********************************************************************************************************************#


def augmented_mccPlot(experiment_choice, depth_choice, user_input):       # 18

    global matthews_set, mcc

    print()
    print("augmented MCC +++++ in data_visualization.py")


    # Creat a barplot of the batches' MCC scores
    sns.set(color_codes=True)

    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, hue=matthews_set,
                     palette="blend:#7AB,#EDA", legend=False)

    ax.hlines(mcc, *ax.get_xlim())
    ax.annotate(f'MCC:\n {mcc:.3f}', xy=(ax.get_xlim()[1], mcc))


    plt.title(f"{experiment_choice}, {depth_choice}, {user_input}, Model ++ MCC PLOT")
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')
    plt.show()

    print(experiment_choice, depth_choice, user_input, "Model ++ MCC PLOT")

#**********************************************************************************************************************#
# Augmented training and validation polt.
#**********************************************************************************************************************#


def augmented_training_and_validation_plot(experiment_choice, depth_choice, user_input):        # 12

    from augmented_training import train_loss, Macro_f1, accuracy, val_loss, text_encoded
    from augmented_metrics import y_preds, preds_output

    print()
    print("")
    print("augmented training and validation plot +++++ in data_visualization.py")
    print()

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
    plt.title(f"{experiment_choice}, {depth_choice}, {user_input}, Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    print(experiment_choice, depth_choice, user_input, " Training & Validation Loss")

#**********************************************************************************************************************#
# Multiclass Confusion Matrix
#**********************************************************************************************************************#


def augmented_confusion_matrix(experiment_choice, depth_choice, user_input):        # 13

    global text_encoded

    print()
    print("augmented confusion matrix +++++ in data_visualization.py")
    print()

    # With the predictions, we can plot the confusion matrix.
    cm = confusion_matrix(text_encoded["val"]['label'], y_preds, normalize="true")

    labels = ['incorrect', 'pass', 'acceptable ', 'correct']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="PuBu", values_format=".2f", colorbar=False)
    # disp.plot(cmap = "Set2", values_format = ".2f", colorbar = False)
    # disp.plot(cmap = "tab10", values_format = ".2f", colorbar = False)

    plt.grid(False)

    plt.title(f"{experiment_choice}, {depth_choice}, {user_input}, Model ++ Confusion Matrix")

    print(experiment_choice, depth_choice, user_input, "Model ++ Confusion Matrix")

