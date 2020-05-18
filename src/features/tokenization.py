from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer

from src.data.make_dataset import download_dataset
from src.features.build_features import preprocessing 


def tokenization():
    '''
    tokenization data
    '''
    data=download_dataset()
    X_train,X_test,y_train,y_test=preprocessing(data)
    count_vect = CountVectorizer()
    print(X_train.content)
    X_train_counts = count_vect.fit_transform(X_train.content)
    X_test_counts =count_vect.transform(X_test.content)

    return X_test_counts,X_test_counts