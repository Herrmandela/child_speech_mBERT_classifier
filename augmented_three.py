#**********************************************************************************************************************#
# #  Augmented Model COMMAND
#**********************************************************************************************************************#

print()
print()
print("|| +++In augmented_three.py - AUGMENTED Model COMMAND +++ Loaded ||")
print()
print()


#**********************************************************************************************************************#
# # GPU- Settings and Imports  ****
#**********************************************************************************************************************#
# Also defined in augmented_load_data.py
PATH = "/Users/ph4533/Desktop/PyN4N/gitN4N/mBERT"
tokenizer = AutoTokenizer.from_pretrained(PATH)

from gpu_settings import *

device_name = gpu_settings3
gpu_settings3()

import augmented_imports

user_input = ""

depth_choice = ""

augmentation_choice = ""

from augmented_load_data import *

# **********************************************************************************************************************#
#   Language Choice Menu
# **********************************************************************************************************************#              # 1


def choose_language():                                # 1
    global user_input

    menu_options = ('english', 'farsi', 'greek', 'multilingual', 'exit')

    while True:
        print()
        print('** MENU ++++++ in augmented_three.py **')
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
# #  Special Validation Set
#**********************************************************************************************************************#
from augmented_validation import augmented_validation
augmented_validation()                                  # 2


#**********************************************************************************************************************#
# # Depth Choice and Training Parameters
#**********************************************************************************************************************#


from augmented_data_prep import *
augmented_data_split()                                  # 3

from augmented_training import *
tokenize(batch)                                         # 4
text_encoder(sentences)                                 # 5

label_id_dictionaries()                                 # 6


def choose_model_depth():

    layer_options = ('vanilla', 'shallow', 'inherent')

    while True:
        print()

        print('** MENU ++++++ in augmented_three.py **')
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


from augmented_metrics import compute_metrics
compute_metrics(pred)                                   # 7

augmented_set_training_args(self)                       # 8 this includes trainer.train()

printerclassback = PrinterCallback()
printerclassback.on_epoch_end()                         # 9
printerclassback.augmented_set_training_args()          # 10

from data_visualization import (augmented_mccPlot,
                                augmented_confusion_matrix,
                                augmented_training_and_validation_plot)

augmented_training_and_validation_plot()                # 11

from augmented_metrics import preds_output
preds_output(trainer)                                   # 12

augmented_confusion_matrix()                            # 13



from functionality import augmented_save_model, augmented_load_model
augmented_save_model()                                  # 14
augmented_load_model()                                  # 15

from sample_sentences import augmented_sample_sentences
augmented_sample_sentences()                            # 16

from mcc_evaluation import augmented_mcc_evaluation
augmented_mcc_evaluation()                              # 17

augmented_mccPlot()                                     # 18



#**********************************************************************************************************************#
# # Augmentation Paradigm Choice
#**********************************************************************************************************************#

from augmentation_paradigms import *

def choose_augmentation_paradigm():                   # 19

    layer_options = ('synonym', 'contextual', 'backtranslation')

    while True:
        print()

        print('** MENU ++++++ in augmented_three.py **')
        print('Synonym - ( * Synonym Embeddings* )')
        print('Contextual - ( * Contextual Embeddings * )')
        print('Backtranslation - ( ! Currently Not Available ! )')

        print()

        augmentation_choice = input('Please choose augmentation paradigm for the synthetic data: ').lower()

        if augmentation_choice in layer_options:
            return augmentation_choice

        else:
            print()
            print('Option not recognized')


augmentation_choice = choose_augmentation_paradigm()

if augmentation_choice == 'synonym':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')
    synonym_augmentation()                                                           # 19
    synonym_evaluation_score()                                                       # 20


elif augmentation_choice == 'contextual':
    print()
    print('Processing...++++++ in augmented_three.py')
    # time.sleep(4)
    print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')
    contextual_embedding_augmentation()                                              # 19
    contextual_evaluation_score()                                                    # 20



# elif augmentation_choice == 'backtranslation':
#     print()
#     print('Processing...++++++ in augmented_three.py')
#     # time.sleep(4)
#     print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')
#     backtranslation_augmentation()


#**********************************************************************************************************************#
# # Samples and MCC of Augmented
#**********************************************************************************************************************#

augmented_load_model()                                                               # 21
augmented_sample_sentences()                                                         # 22

from mcc_evaluation import augmented_mcc_evaluation

augmented_mcc_evaluation()                                                           # 23

augmented_mccPlot()                                                                  # 24

