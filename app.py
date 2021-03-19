import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from heapq import nlargest
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
import liwc
import altair as alt
#import spacy
#nlp = spacy.load("en_core_web_sm")


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)
def parseLIWC(x):
    gettysburg_tokens = tokenize(x)
    gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))
    return gettysburg_counts




def load_data():
    DATA_URL = "data/kaggle_train.csv"
    data = pd.read_csv(DATA_URL)
    data = data[~data['text'].isnull()]
    data = data[[ 'title', 'author', 'text', 'label']]
    data['Category'] = data['label'].apply(lambda x: "Real News" if x == 1 else "Fake News")
    return data
    
def app():
    image = Image.open('static/24176744.jpg')
    st.image(image)
    st.title('Fake News Gauge')
    
    data = load_data();
    # present data
    st.sidebar.subheader("Show random news")
    if not st.sidebar.checkbox("Hide",True, key='1'):
        random_news = st.sidebar.radio('Category',("Real News","Fake News"))
        news_article = data.query('Category == @random_news')[['text']].sample(n=1).iat[0,0]

        st.sidebar.markdown(news_article.strip())
    
    # viz
    df_flat = pd.read_csv('data/flaten_file2.csv')
    st.sidebar.subheader("Data Analysis")
    if not st.sidebar.checkbox("Hide",True, key='2'):
        select = st.sidebar.selectbox('Feature Extraction: ',['Part-of-Speech','LIWC'], key='3')
        if select == "Part-of-Speech":
            st.markdown(""" Fake news contains more function words than Real News; Real news contains more content words. """)
            pos_df = df_flat.loc[:, (df_flat.columns.str.startswith('pos.')|df_flat.columns.str.startswith('label'))]
            pos_grouped = pos_df.groupby('label').sum().T
            # 0:10387
            # 1:10374
            pos_grouped[0] = pos_grouped[0]/10387
            pos_grouped[1] = pos_grouped[1]/10374
            POS_Tags = ['pos.ADJ','pos.PROPN','pos.NOUN','pos.PUNCT','pos.ADP','pos.DET']
            pos_grouped_chart_d = pos_grouped.T[POS_Tags]
            fig = go.Figure(data=[
                go.Bar(name='True News', x=POS_Tags, y=list(pos_grouped_chart_d.iloc[0])),
                go.Bar(name='False News', x=POS_Tags, y=list(pos_grouped_chart_d.iloc[1]))
            ])
            fig.update_layout(yaxis_title="Frequency",xaxis_title="Part of Speech Tags");
            st.plotly_chart(fig)
        elif select == "LIWC":
            st.markdown(""" Fake news contains more function words than Real News; Real news contains more content words. """)
            liwc_df = df_flat.loc[:, (df_flat.columns.str.startswith('liwc')|df_flat.columns.str.startswith('label'))]
            liwc_grouped = liwc_df.groupby('label').sum().T
            # 0:10387
            # 1:10374
            liwc_grouped[0] = liwc_grouped[0]/10387
            liwc_grouped[1] = liwc_grouped[1]/10374
            LIWC_Tags = ['liwc_ct.Certain', 'liwc_ct.Informal', 'liwc_ct.Cause','liwc_ct.Adj', 'liwc_ct.Prep', 'liwc_ct.Function']
            liwc_grouped_chart_d = liwc_grouped.T[LIWC_Tags]
            fig = go.Figure(data=[
                go.Bar(name='True News', x=LIWC_Tags, y=list(liwc_grouped_chart_d.iloc[0])),
                go.Bar(name='False News', x=LIWC_Tags, y=list(liwc_grouped_chart_d.iloc[1]))
            ])
            fig.update_layout(yaxis_title="Frequency",xaxis_title="LIWC Tags");
            st.plotly_chart(fig)
    
    st.sidebar.subheader("Model Selection")
    if not st.sidebar.checkbox("Hide",True, key='4'):
        select = st.sidebar.selectbox('Feature Extraction: ',['Part-of-Speech','LIWC'], key='7')
        st.markdown("Parameter Table")
        parameter_table = pd.DataFrame([[10, 1, 0.1, 0.01, 0.001],[1.00000000e-05, 3.16227766e-04, 1.00000000e-02, 3.16227766e-01,1.00000000e+01],[1.00000000e-05, 3.16227766e-04, 1.00000000e-02, 3.16227766e-01,1.00000000e+01],[1, 2, 5, 10, 50],
[10, 1, 0.1, 0.01, 0.001]], columns=['p1','p2','p3','p4','p5'],index=[ "Naive Bayes - alpha", "Linear SVM - C", "Logistic Regression - C", "Random Forest - max depth", "Multilayer Perceptron - alpha"])
        st.write(parameter_table)
        
        if select == 'Part-of-Speech':
            pos_result = pd.read_csv('data/pos_result_flatten.csv')
            fig = px.line(pos_result, x="parameter", y="acuracy", color='model')
            st.plotly_chart(fig)
        elif select == "LIWC":
            liwc_result = pd.read_csv('data/liwc_result_flatten.csv')
            fig = px.line(liwc_result, x="parameter", y="acuracy", color='model')
            st.plotly_chart(fig)
    
    
    st.sidebar.subheader("Predicting Model")
    if not st.sidebar.checkbox("Hide",False, key='5'):
        testNews = st.text_input("News goes here")
        parse, category_names = liwc.load_token_parser('data/queryDictionary.dic')
        # when 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"): 
            pickle_in = open('classifier.pkl', 'rb') 
            classifier = pickle.load(pickle_in)
            x_parsed = parseLIWC(testNews)
            liwc_ct = [x_parsed[name.replace("liwc_ct.","")] for name in liwc_df.columns[1:]]
            liwc_sum = sum(liwc_ct)
            liwc_ip = [ (xi*1.0)/liwc_sum for xi in liwc_ct]
            
            prediction = classifier.predict([liwc_ip])
            if prediction == 1:
                result = "True"
            else:
                result = "False"
            st.success('The news might be {}'.format(result))
    
 

    
    st.sidebar.subheader("Methology")
    if not st.sidebar.checkbox("Hide",True, key='6'):
        st.markdown(""" ## Methology
         * This model trained by Politifact News Source
         * Utilizes POS and LIWC to extract features
         * And cross-validate trained with different classification models
    """)
        st.write("GitHub: [ctsuiyao/202101_fake_news_gauge](https://github.com/ctsuiyao/202101_fake_news_gauge)")


if __name__ == '__main__':
    app()