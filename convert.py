import pandas as pd
import numpy as np
import os

data_path = '/mnt/hdd1_part1/UROP22_winter/intern_6/data/new_csv'
filename_path = 'dataset/meta/ours/dev.csv'

def main():
    file_lists = pd.read_csv(filename_path)['filename']
    for fn in file_lists:
        path = os.path.join(data_path, str(fn)+'.csv')
        print(fn)
        df = pd.read_csv(path, names=['class','st','et','azi','ele'], dtype={'class':np.int64, 'st':np.float64, 'et':np.float64, 'azi':np.int64, 'ele':np.int64})
        
        df['st'] = (df['st']*10).astype(np.int32)
        df['et'] = (df['et']*10).astype(np.int32)
        
        outputs = []
        for i in np.arange(len(df)):
            class_idx = df['class'][i]
            azi = df['azi'][i] 
            if azi >= 180:
                azi -=360
            elif azi <-180:
                azi += 360
            ele = df['ele'][i]
            
            for i, iframe in enumerate(np.arange(df['st'][i],df['et'][i]+1)):                
                if iframe>299:
                    break
                outputs.append([iframe, class_idx, 0, azi, ele])
        split = path.split('/')
        split[-2] = 'metadata_dev'
        path = "/".join(split)
        
        file_df = pd.DataFrame(outputs).sort_values(0, ascending=True)
        file_df.to_csv(path, index=False, header=False)

if __name__ == '__main__':
    main()
