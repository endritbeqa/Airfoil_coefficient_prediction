import os
import math
import shutil

def read_file(file_path):
    
    file = open(file_path, 'r')
    rows = file.readlines() 
    entries = []

    for idx, row in enumerate(rows):
        label , loss = row.split(':')
        label = label[2:-1]
        loss = float(loss[1:])
        entries.append([label,loss])

    return entries 


def calculate_outliers(data):

    outliers_labels = []

    for entry in data:
        if math.isnan(entry[1]) or entry[1]>1:
            outliers_labels.append(entry[0])
    
    return outliers_labels


def move_corrupted_examples(corrupt_files):
    
    data_dir = '/naca_transformer/Bluff_data/BluffFOAM-Raw'
    corrupt_data_dir = '/naca_transformer/Bluff_data/corrupted_files'

    for label in corrupt_files:
        try: 
            shutil.move(os.path.join(data_dir,'stl',label+'.stl'),os.path.join(corrupt_data_dir,'stl',label+'.stl'))
            shutil.move(os.path.join(data_dir,'vtu',label+'.vtu'),os.path.join(corrupt_data_dir,'vtu',label+'.vtu'))
        except:
            print("Already removed")


if __name__=='__main__':

    losses_training = '/local/disk1/ebeqa/naca_transformer/outputs/losses/losses_training.txt'
    losses_testing = 'naca_transformer/outputs/losses/losses_testing.txt'

    for path in [losses_training, losses_testing]:
        data = read_file(path)
        outliers = calculate_outliers(data)
        move_corrupted_examples(outliers)


