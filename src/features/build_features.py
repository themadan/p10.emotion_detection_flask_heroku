from nltk.corpus import stopwords
from textblob import Word
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('wordnet')


def preprocessing(data):
    '''
    preprocess data
    '''
    # Remove all the new line character
    data['content'] = data['content'].str.replace('\n', '')

    #Replace full stop with blank
    data['content'] = data['content'].str.replace('.', '')

    #Remove irrelevant characters other than alphanumeric and space 
    data['content']=data['content'].str.replace('[^A-Za-z0-9\s]+', '')

    #Remove links from the text
    data['content']=data['content'].str.replace('http\S+|www.\S+', '', case=False)

    #Convert everything to lowercase
    data['content']=data['content'].str.lower()

    #Removing Punctuation, Symbols
    data['content'] = data['content'].str.replace('[^\w\s]',' ')

    #Assign target variable
    target=data.emotion
    data = data.drop(['emotion'],axis=1)

    #label encoder
    le=LabelEncoder()
    target=le.fit_transform(target)

    #Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.4, random_state=42)
    
    print('helo iam madan')

    return X_train,X_test,y_train,y_test
