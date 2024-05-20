#**********************************************************************************************************************#
# #  Augmented Model COMMAND
#**********************************************************************************************************************#

print()
print()
print("|| +++ AUGMENTED Model COMMAND +++ Loaded ||")
print()
print()


#**********************************************************************************************************************#
# # GPU- Settings and Imports  ****
#**********************************************************************************************************************#
# Also defined in augmented_load_data.py

from gpu_settings import *

device_name = gpu_settings2
gpu_settings2()

import augmented_imports
from menus import experiment_choice


user_input = ""

depth_choice = ""

aug_or_ex = ""

augmentation_choice = ""

# **********************************************************************************************************************#
#   Language Choice Menu
# **********************************************************************************************************************#              # 1
from augmented_load_data import *

def choose_language():
    global user_input

    menu_options = ('english', 'farsi', 'greek', 'multilingual', 'exit')

    while True:
        print()
        print('** MENU **')
        print('- English')
        print('- Farsi')
        print('- Greek')
        print('- Multilingual')
        print('- Exit')

        print()
        user_input = input('Please choose language model: ').lower()

        if user_input in menu_options:
            return user_input

        else:
            print()
            print('Option not recognized')


user_input = choose_language()

if user_input == 'english':
    print('Processing...')
    #time.sleep(3)
    print('You choose English')
    augmented_load_english()

elif user_input == 'farsi':
    print('Processing...')
    #time.sleep(3)
    print('You choose Farsi')
    augmented_load_farsi()

elif user_input == 'greek':
    print('Processing...')
    #time.sleep(3)
    print('You choose Greek')
    augmented_load_greek()

elif user_input == 'multilingual':
    print('Processing...')
    #time.sleep(3)
    print('You choose the Multilingual Data')
    augmented_load_all()


elif user_input == 'exit':
    print()
    print(' Good Bye!')
    exit()

#**********************************************************************************************************************#
# # Depth Choice and Training Parameters
#**********************************************************************************************************************#
import augmented_data_prep
                                                                 #                                # 6
def choose_model_depth():

    layer_options = ('vanilla', 'shallow', 'inherent')

    while True:
        print()

        print('** MENU **')
        print('Vanilla - ( * No layers Frozen * )')
        print('Shallow - ( * Shallow Layers Frozen + Output Layer * )')
        print('Inherent - ( * All Layers Frozen Except for Input and Output Layers * )')

        print()

        depth_choice = input('Please choose model depth: ').lower()

        if depth_choice in layer_options:
            return depth_choice

        else:
            print()
            print('Option not recognized')


depth_choice = choose_model_depth()

from augmented_training import *                       # 8 this includes trainer.train()

if depth_choice == 'vanilla':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')
    augmented_training_vanilla()

elif depth_choice == 'shallow':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')
    augmented_training_shallow()

elif depth_choice == 'inherent':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')
    augmented_training_inherent()




from augmented_datavis import *
#preds_output(trainer)

print("Augmented Training and Validation Plot")
augmented_training_and_validation_plot(experiment_choice, depth_choice, user_input)                # 11

print("Augmented Confusion Matrix")
augmented_confusion_matrix(experiment_choice, depth_choice, user_input)                            # 13



from functionality import augmented_save_model, augmented_load_model
augmented_save_model()                                  # 14
augmented_load_model(output_dir)                                  # 15

from sample_sentences import augmented_sample_sentences
augmented_sample_sentences(experiment_choice, depth_choice, user_input)                            # 16

from mcc_evaluation import augmented_mcc_evaluation
augmented_mcc_evaluation()                              # 17

augmented_mccPlot(experiment_choice, depth_choice, user_input)                                     # 18

print("Augmented Confusion Matrix")
augmented_confusion_matrix(experiment_choice, depth_choice, user_input)                            # 13


#**********************************************************************************************************************#
# # Augmentation Paradigm Choice
#**********************************************************************************************************************#

def choose_augmentation_paradigm():                   # 19

    layer_options = ('synonym', 'contextual', 'backtranslation', 'exit')

    while True:
        print()

        print('** MENU **')
        print('Synonym - ( * Synonym Embeddings* )')
        print('Contextual - ( * Contextual Embeddings * )')
        print('Backtranslation - ( ! Currently Not Available ! )')
        print('Exit')

        print()

        augmentation_choice = input('Please choose augmentation paradigm for the synthetic data: ').lower()

        if augmentation_choice in layer_options:
            return augmentation_choice

        else:
            print()
            print('Option not recognized')


augmentation_choice = choose_augmentation_paradigm()

from augmentation_paradigms import *

if augmentation_choice == 'synonym':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')
    synonym_evaluation_score()                                                       # 20


elif augmentation_choice == 'contextual':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')
    contextual_evaluation_score()                                                    # 20

elif augmentation_choice == 'exit':
    print()
    print(' Good Bye!')
    exit()

# elif augmentation_choice == 'backtranslation':
#     print()
#     print('Processing...++++++ in augmented_three.py')
#     # time.sleep(4)
#     print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')
#     backtranslation_augmentation()


#**********************************************************************************************************************#
# # Samples and MCC of Augmented
#**********************************************************************************************************************#
experiment_choice = experiment_choice
augmented_load_model(output_dir)
augmented_sample_sentences(experiment_choice, depth_choice, user_input)

augmented_mcc_evaluation()                                                           # 23
augmented_mccPlot(experiment_choice, depth_choice, user_input)                       # 24
