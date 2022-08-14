import streamlit as st
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import re
import string
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from csv import writer
import pickle
import string
warnings.filterwarnings('ignore')







def text_clean_1(text):
    text=str(text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_clean_2(text):
    text = str(text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text
def countPlot():
    data=pd.read_csv('Data/train.csv')
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x = data.Is_Response)
    st.pyplot(fig)



def main():

    st.title('Hotel Review Classifier')
    sentence = str(st.text_input('Feedback:'))
    if st.button("submit"):

        eg = [sentence]
        Reviewdata = pd.read_csv('Data/train.csv')
        id1 = (len(Reviewdata['User_ID']) - 2 + 10327) + 1
        browser = "Chrome"
        Device = 'Desktop'

        Reviewdata.drop(columns = ['User_ID', 'Browser_Used', 'Device_Used'], inplace = True)
        cleaned1 = lambda x: text_clean_1(x)
        Reviewdata['cleaned_description'] = pd.DataFrame(Reviewdata.Description.apply(cleaned1))
        cleaned2 = lambda x: text_clean_2(x)
        Reviewdata['cleaned_description_new'] = pd.DataFrame(Reviewdata['cleaned_description'].apply(cleaned2))

        Independent_var = Reviewdata.cleaned_description_new
        Dependent_var = Reviewdata.Is_Response
        IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size=0.1, random_state=225)
        tvec = TfidfVectorizer()
        clf2 = LogisticRegression(solver="lbfgs")
        model = Pipeline([('vectorizer', tvec), ('classifier', clf2)])
        model.fit(IV_train, DV_train)
        predictions = model.predict(IV_test)





        res = model.predict(eg)
        List = [id1, eg, browser, Device, res[0]]
        with open('Data/train.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()



        df1 = pd.read_csv('Data/train.csv')
        df1.tail(3)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Positive")
            st.markdown(len(df1[df1.Is_Response=='positive']))
        with col2:
            st.header("Negative")
            st.markdown(len(df1[df1.Is_Response=='negative']))

        countPlot()

        pickle_mod = open("Hotelsentiment.pkl", mode="wb")
        pickle.dump(model,pickle_mod)
        pickle_mod.close()







if __name__ == '__main__':
	main()


