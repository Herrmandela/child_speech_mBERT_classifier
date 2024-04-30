import time

print()
print()
print("|| +++In menus.py +++ Loaded ||")
print()
print()

#**********************************************************************************************************************#
# # Load - English, Farsi, Greek and Multilingual - Training Data  ****
#**********************************************************************************************************************#

print()
print("+++++++In menus.py+++++++")
print("Loading training data in menus.py")
print()

#**********************************************************************************************************************#
# #  Experiment Choice Menu (1 - Binary Classification, 2 - Multi-Class, 3 - Augmentation)
#**********************************************************************************************************************#

from load_paradigm import *

experiment_choice = ""

def choose_experiment():

    experiment_options = ('binary', 'multiclass', 'augmented', 'exit')

    while True:
        print()
        print('** MENU +++++ in menus.py **')
        print('- Binary (for TOLD only')
        print('- Multiclass (for TOLD/CELF)')
        print('- Augmented (CELF)')
        print('- Exit')

        print()
        experiment_choice = input('Please choose language model: ').lower()

        if experiment_choice in experiment_options:
            return experiment_choice

        else:
            print()
            print('Experiment Choice not recognized')


experiment_choice = choose_experiment()

if experiment_choice == 'binary':
    print('Processing...++++++  in menus.py')
    #time.sleep(3)
    print('You choose the Binary Paradigm')
    load_binary()

elif experiment_choice == 'multiclass':
    print('Processing...++++++  in menus.py')
    #time.sleep(3)
    print('You choose the Multi-Class Paradigm')
    load_multiclass()

elif experiment_choice == 'augmented':
    print('Processing...++++++  in menus.py')
    #time.sleep(3)
    print('You choose the Augmented Paradigm')
    load_augmented()

elif experiment_choice == 'exit':
    print()
    print(' Good Bye!')
    exit()


print("experiment_choice is:", experiment_choice)
#**********************************************************************************************************************#
# #  Language Choice Menu
#**********************************************************************************************************************#

#from load_data import *                     # Load_Data is numbers 5, 6, 7, 8, 9, 10, 11

#
# def choose_language():
#     menu_options = ('english', 'farsi', 'greek', 'multilingual', 'exit')
#
#     while True:
#         print()
#         print('** MENU ++++++  in menus.py **')
#         print('- English')
#         print('- Farsi')
#         print('- Greek')
#         print('- Multilingual')
#         print('- Exit')
#
#         print()
#         user_input = input('Please choose language model: ').lower()
#
#         if user_input in menu_options:
#             return user_input
#
#         else:
#             print()
#             print('Option not recognized')
#
#
# user_input = choose_language()
#
# if user_input == 'english':
#     print('Processing...++++++  in menus.py')
#     #time.sleep(3)
#     print('You choose English')
#     load_english
#
# elif user_input == 'farsi':
#     print('Processing...++++++  in menus.py')
#     #time.sleep(3)
#     print('You choose Farsi')
#     load_farsi
#
# elif user_input == 'greek':
#     print('Processing...++++++  in menus.py')
#     #time.sleep(3)
#     print('You choose Greek')
#     load_greek
#
# elif user_input == 'multilingual':
#     print('Processing...++++++  in menus.py')
#     #time.sleep(3)
#     print('You choose the Multilingual Data')
#     load_all
#
#
# elif user_input == 'exit':
#     print()
#     print(' Good Bye!')
#     exit()

#**********************************************************************************************************************#
# # Functionality
#**********************************************************************************************************************#
#
# import model_tokenizer   # numbers 12, 13, 14, 15, 16
#
#
# from functionality import *
# trainingAndValidation()
# saveMemory()
# #

#**********************************************************************************************************************#
# # Model Depth Menu
#**********************************************************************************************************************#
#
#
# from training import *
                                                # numbers 19
#
# def choose_model_depth():
#     layer_options = ('vanilla', 'shallow', 'inherent')
#
#     while True:
#         print()
#
#         print('** MENU ++++++  in menus.py **')
#         print('Vanilla - ( * No layers Frozen * )')
#         print('Shallow - ( * Shallow Layers Frozen + Output Layer * )')
#         print('Inherent - ( * All Layers Frozen Except for Input and Output Layers * )')
#
#         print()
#         depth_choice = input('Please choose model depth: ').lower()
#
#         if depth_choice in layer_options:
#             return depth_choice
#
#         else:
#             print()
#             print('Option not recognized')
#
# depth_choice = choose_model_depth()
#
# if depth_choice == 'vanilla':
#     print()
#     print('Processing...++++++  in menus.py')
#     #time.sleep(4)
#     print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')
#     trainingVanilla
#
# elif depth_choice == 'shallow':
#     print()
#     print('Processing...++++++  in menus.py')
#     #time.sleep(4)
#     print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')
#     trainingShallow
#
# elif depth_choice == 'inherent':
#     print()
#     print('Processing...++++++  in menus.py')
#     #time.sleep(4)
#     print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')
#     trainingInherent
#
#
#
#
#
# optimizer()                             # number 20
#
# linear_schedule()                       # number 21
#
#
# flat_accuracy()                         # number 22
# format_time()                           # number 23
#
#
# train()                                 # number 24
#
# training_summary()                      # number 25

# # *******************************************************
# # VIS
# # *******************************************************
# from data_visualization import *
# plotValidationAndLoss()                 # number 27
#
# from mcc_data_prep import mcc_data_prep
# mcc_data_prep()                         # number 28
#
# from mcc_evaluation import mcc_evaluation
# mcc_evaluation()                        # number 29
# plotMCC()                               # number 30
#
# # *******************************************************
# # OUT
# # *******************************************************
#
# output_dir = save_model()
# save_model()                           # numbers 31
#
# #print(output_dir)
#
# load_model(output_dir)                            # number 32
#
# from sample_sentences import *
#
# sampleSentences()                       # number 33
