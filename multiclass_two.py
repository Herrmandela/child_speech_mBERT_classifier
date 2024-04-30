#**********************************************************************************************************************#
# #  MultiClass Model COMMAND
#**********************************************************************************************************************#
print()
print()
print("|| +++ multiclass_two.py - MultiClass Model COMMAND +++ Loaded ||")
print(" load from dependencies.txt")
print(" load imports.py")

#**********************************************************************************************************************#
# # GPU- Settings and Imports  ****
#**********************************************************************************************************************#

PATH = "/Users/ph4533/Desktop/PyN4N/gitN4N/mBERT"
tokenizer = AutoTokenizer.from_pretrained(PATH)

from gpu_settings import *

device_name = gpu_settings1
gpu_settings1()                             # number 3

device = gpu_settings2
gpu_settings2()                             # number 4

import multiclass_imports
multiclass_imports()
#**********************************************************************************************************************#
# #  MultiClass Language Choice Menu
#**********************************************************************************************************************#

user_input = ""

depth_choice = ""

experiment_choice = ""

from multiclass_load_data import *          # 1 this together with Label assignment

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



#**********************************************************************************************************************#
# # Depth Choice and Training Parameters
#**********************************************************************************************************************#

from multiclass_training import *
training_parameters()               # 2 in multiclass_training.py

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

depth_choice = choose_model_depth()

if depth_choice == 'vanilla':
    print()
    print('Processing...++++++ in multiclass_two.py ')
    #time.sleep(4)
    print('You chose the Vanilla Model, All LAYERS WILL UNDERGO TRAINING')
    multiclass_training_vanilla

elif depth_choice == 'shallow':
    print()
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(4)
    print('You chose the Shallow Model, ONLY THE INITIAL LAYERS WILL UNDERGO TRAINING')
    multiclass_training_shallow

elif depth_choice == 'inherent':
    print()
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(4)
    print('You chose the Inherent Model, NO LAYERS WILL UNDERGO TRAINING')
    multiclass_training_inherent


from multiclass_data_prep import *

multiclass_data_split()                     # number 4

multiclass_data_pandafication()             # number 5

preprocess_function()                       # number 6


#**********************************************************************************************************************#
# # Language Choice for Dataset Evaluation # number 7
#**********************************************************************************************************************#

user_input = user_input

if user_input == 'english' or user_input == 'multilingual':
    print('Processing...++++++ in multiclass_two.py')
    #time.sleep(3)
    print('You choose English')
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

#**********************************************************************************************************************#
# #
#**********************************************************************************************************************#


from multiclass_metrics import *


get_preds_from_logits(logits)           # 8

compute_metrics(eval_pred)              # 9

multitaskCT = MultiTaskClassificationTrainer(Trainer)  # 10
multitaskCT.compute_loss(self, model, inputs, return_outputs=False)   # 11


from multiclass_training import *
printerclassback = PrinterCallback(TrainerCallback)        # 12
printerclassback.on_epoch_end(self, args, state, control, logs=None, **kwargs)     # 13
                                                                                   # Check whether this
                                                                                   # one executes automatically
                                                                                   # with the initiation of the class

printerclassback.multiclass_set_training_args()         # 14

trainer.train()         # 15

trainer.evaluate()      # 16

from data_visualization import multiclass_confusion_matrix, multiclass_training_and_validation_plot
multiclass_confusion_matrix(experiment_choice, depth_choice, user_input)              # 17

multiclass_training_and_validation_plot(experiment_choice, depth_choice, user_input)  # 18

from sample_sentences import multiclass_sample_sentences
multiclass_sample_sentences()                                   # 19

from mcc_evaluation import *
multiclass_mcc_evaluation()                                     # 20

from data_visualization import plotMCC
plotMCC(experiment_choice, depth_choice, user_input)            # 21

from functionality import multiclass_save_model, multiclass_load_model

output_dir = multiclass_save_model()                            # 22

multiclass_load_model(output_dir)                               # 23











