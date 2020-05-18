import pandas as pd


def download_dataset():
    '''
    download datasets
    '''

    columns=['emotion','content']
    data = pd.read_csv('https://github.com/themadan/p8.emotion-detection/blob/master/ISEAR.csv',names=columns)

    return data