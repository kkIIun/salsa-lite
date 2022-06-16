from fileinput import filename
import pandas as pd
import numpy as np
import os

data_path = '/mnt/hdd1_part1/UROP22_winter/intern_6/data/new_csv'

def main():
    for split in ['train','val','test']:
        files = os.listdir(data_path)
        length = len(files)
        train, val, test = length*0.6, length*0.2, length*0.2
        
        train_file_lists = []
        val_file_lists = []
        test_file_lists = []
        for i,f in enumerate(files):
            if i < train:
                train_file_lists.append(os.path.splitext(f)[0])
            elif i < val+train:
                val_file_lists.append(os.path.splitext(f)[0])
            else:
                test_file_lists.append(os.path.splitext(f)[0])

        test_filename_df = pd.DataFrame({'filename':train_file_lists + val_file_lists + test_file_lists})
        test_filename_df.to_csv('dev.csv', index=False)

        test_filename_df = pd.DataFrame({'filename':train_file_lists})
        test_filename_df.to_csv('train.csv', index=False)

        test_filename_df = pd.DataFrame({'filename':val_file_lists})
        test_filename_df.to_csv('val.csv', index=False)

        test_filename_df = pd.DataFrame({'filename':test_file_lists})
        test_filename_df.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()
