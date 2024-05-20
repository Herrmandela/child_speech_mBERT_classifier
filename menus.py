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

