import matplotlib.pyplot as plt
import seaborn as sns

print()
print("+++++++In data_visualization.py+++++++")

#**********************************************************************************************************************#
# Plot and Validation after training
#**********************************************************************************************************************#

def plotValidationAndLoss(user_input, depth_choice, experiment_choice):

    from training import df_stats

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label='Training')
    plt.plot(df_stats['Valid. Loss'], 'g-o', label='Validation')

    # Label the plot
    plt.title(f"{experiment_choice} {depth_choice} {user_input} Model - Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5])

    plt.show()
    

    # print(experiment_choice, depth_choice, user_input, 'Training & Validation Loss')

#**********************************************************************************************************************#
# Build MCC
#**********************************************************************************************************************#


def plotMCC(user_input, depth_choice, experiment_choice):

    from mcc_evaluation import matthews_set, mcc

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
