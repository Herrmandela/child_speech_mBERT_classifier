#**********************************************************************************************************************#
# #  MultiClass Model COMMAND
#**********************************************************************************************************************#

print()
print("|| +++ multiclass_two.py - MultiClass Model COMMAND +++ Loaded ||")
print()
print()

#**********************************************************************************************************************#
# # GPU- Settings and Imports  ****
#**********************************************************************************************************************#

from gpu_settings import *

device = gpu_settings1
gpu_settings1()                             # number 4

import multiclass_imports

#device = gpu_settings3
#gpu_settings3()

print(device)
from augmented_load_data import All_text
#**********************************************************************************************************************#
# #  MultiClass Language Choice Menu
#**********************************************************************************************************************#

user_input = "english"

depth_choice = "vanilla"

experiment_choice = "multiclass"


from multiclass_load_data import *          # 1 this together with Label assignment


multiclass_load_english()       #

from multiclass_data_prep import *   #

dataset_test_evaluation_english()       #

import multiclass_metrics, multiclass_training, multiclass_datavis, sample_sentences

multiclass_training.multiclass_training_vanilla()

multiclass_datavis.multiclass_training_and_validation_plot(experiment_choice, depth_choice, user_input)

sample_sentences.multiclass_sample_sentences()

import mcc_data_prep
#import mcc_evaluation

multiclass_datavis.multiclass_plotMCC(experiment_choice, depth_choice, user_input)

"""
def choose_language():
    global user_input

    menu_options = ('english', 'farsi', 'greek', 'multilingual', 'exit')


    while True:
        print()
        print('** MENU ++++++ in multiclass_two.py **')
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
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose English')
    multiclass_load_english()

elif user_input == 'farsi':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose Farsi')
    multiclass_load_farsi()

elif user_input == 'greek':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose Greek')
    multiclass_load_greek()

elif user_input == 'multilingual':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose the Multilingual Data')
    multiclass_load_all()


elif user_input == 'exit':
    print()
    print(' Good Bye!')
    exit()




#from multiclass_data_prep import *

#multiclass_data_split()
#multiclass_data_pandafication()

#**********************************************************************************************************************#
# # Language Choice for Dataset Evaluation # number 7
#**********************************************************************************************************************#

user_input = user_input
print(user_input)

if user_input == 'english' or user_input == 'multilingual':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('+++++++++++++++You choose English++++++++++DS++++++')
    dataset_test_evaluation_english()           # 7

elif user_input == 'farsi':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose Farsi')
    dataset_test_evaluation_farsi()             # 7

elif user_input == 'greek':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose Greek')
    dataset_test_evaluation_greek()             # 7

#from MCtrainer import *
#**********************************************************************************************************************#
# # Depth Choice and Training Parameters
#**********************************************************************************************************************#
import multiclass_metrics, multiclass_training, multiclass_datavis, sample_sentences
def choose_model_depth():           # 3 in multiclass_training.py
    global depth_choice

    layer_options = ('vanilla', 'shallow', 'inherent')


    while True:
        print()

        print('** MENU ++++++ in multiclass_two.py **')
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

print("ficeerts")
depth_choice = choose_model_depth()
print("ficeerts2")

if depth_choice == 'vanilla':
    print()
    print('Processing...++++++ in multiclass_two.py ')
    #time.sleep(4)
    print("ficeerts3")
    multiclass_training.multiclass_training_vanilla()
    print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')


elif depth_choice == 'shallow':
    print()
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(4)
    multiclass_training.multiclass_training_shallow()
    print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')


elif depth_choice == 'inherent':
    print()
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(4)
    multiclass_training.multiclass_training_inherent()
    print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')


print("!!!!!!!!!!!!!loading MCtrainer training_parameters!!!!!!!!!!!!!!!!!!!")



multiclass_datavis.multiclass_training_and_validation_plot(experiment_choice, depth_choice, user_input)

sample_sentences.multiclass_sample_sentences()

import mcc_data_prep
#import mcc_evaluation

multiclass_datavis.multiclass_plotMCC(experiment_choice, depth_choice, user_input)

"""









