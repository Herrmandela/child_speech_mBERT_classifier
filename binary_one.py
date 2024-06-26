#**********************************************************************************************************************#
# #  Binary Model COMMAND - 20 Mansksteps in total.
#**********************************************************************************************************************#

print()
print()
print("|| +++ Binary Model COMMAND +++ Loaded ||")
print()
print()

#**********************************************************************************************************************#
# # GPU- Settings and imports   ****
#**********************************************************************************************************************#

from gpu_settings import *

#device_name = gpu_settings1
#gpu_settings1()

device = gpu_settings2
gpu_settings2()

print(device)
import imports
#**********************************************************************************************************************#
# #  Language Choice Menu
#**********************************************************************************************************************#

user_input = ""

depth_choice = ""

experiment_choice = ""

from binary_load_data import *

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
    load_english()

elif user_input == 'farsi':
    print('Processing...')
    #time.sleep(3)
    print('You choose Farsi')
    load_farsi()

elif user_input == 'greek':
    print('Processing...')
    #time.sleep(3)
    print('You choose Greek')
    load_greek()

elif user_input == 'multilingual':
    print('Processing...')
    #time.sleep(3)
    print('You choose the Multilingual Data')
    load_all()


elif user_input == 'exit':
    print()
    print(' Good Bye!')
    exit()


#**********************************************************************************************************************#
# # Functionality
#**********************************************************************************************************************#

import model_tokenizer      # 4

print("user_input is:", user_input)


from functionality import *
trainingAndValidation()     # 5
saveMemory()                # 6


#**********************************************************************************************************************#
# # Binary Model Depth Menu
#**********************************************************************************************************************#

from training import trainingVanilla, trainingShallow, trainingInherent      # 7


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

if depth_choice == 'vanilla':
    print()
    print('Processing...')
    #time.sleep(4)
    print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')
    trainingVanilla()

elif depth_choice == 'shallow':
    print()
    print('Processing...')
    #time.sleep(4)
    print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')
    trainingShallow()

elif depth_choice == 'inherent':
    print()
    print('Processing...')
    #time.sleep(4)
    print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')
    trainingInherent()

print("depth choice is:", depth_choice)




from training import optimizer_and_scheduler, train, training_summary

optimizer_and_scheduler()                             # 8 in training.py

print()
print()
print("!!! Ran Optimizer !!!")
print()
print()


train()                                 # 12 in training.py

training_summary()                      # 13 in training.py


# *******************************************************
# VIS
# *******************************************************
from binary_datavis import plotValidationAndLoss, plotMCC
plotValidationAndLoss(user_input, depth_choice, experiment_choice)                 # 14 in datavisualization.py
                                                                 # 15 in mcc_data_prep.py

from mcc_evaluation import *
binary_mcc_evaluation()                                                        # 16 in mcc_evaluation
plotMCC(user_input, depth_choice, experiment_choice)                               # 17 in datavisualization.py

# *******************************************************
# OUT
# *******************************************************

output_dir = save_model()
#save_model()                                     # 18 in functionality.py

#print(output_dir)

from sample_sentences import input_texts

load_model(output_dir)                            # 19 in functionality.py

#sampleSentences()                                 # 20 in functionlity.py