import pandas as pd
import plotly.express as px
import streamlit as st
import pickle
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_score,f1_score,accuracy_score
import seaborn as sns
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def display_dataframe_info(data):
    #st.subheader('Dataframe')
    n, m = data.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)
    st.dataframe(data.head(10))

def target_column_analysis(data):
    st.subheader('Target column Analysis:')
    fig = px.pie(data, names='rul_class')
    c1, c2, c3 = st.columns([0.5, 2, 0.5])
    c2.plotly_chart(fig)

def basic_statistics(data):
    st.subheader('Basic Statistics:')
    st.write(data.describe())



def correlation_matrix(data):
    st.subheader('Correlation Matrix:')
    corr = data.corr()

    fig = px.imshow(corr, color_continuous_scale='BuGN', width=1000, height=1000)
    
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            fig.add_annotation(
                x=j,
                y=i,
                text=f'{corr.iat[i, j]:.2f}',
                showarrow=False,
                font=dict(color="black")
            )

    st.plotly_chart(fig)

def numerical_data_analysis(data, column):
    fig = px.histogram(data, x=column)
    
    return fig

def box_plot(data, column):
    fig = px.box(data, y=column,width=800,height=800,)
    
    return fig

def feature_importance_train():
    st.subheader('Feature Importance')
    st.image('new.png', caption='Feature Importance', width=800)




def load_model(model_path):
    try:
        with open(model_path , 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def predict(model, data):
    return model.predict(data)


    return model.predict_proba(data)[:, 0]
def likelihood0(model, data):
    return [f"{prob*100:.2f}" for prob in model.predict_proba(data)[:, 0]]



def likelihood1(model,data):
    return [f"{prob*100:.2f}" for prob in model.predict_proba(data)[:, 1]]

def likelihood2(model,data):
    return [f"{prob*100:.2f}" for prob in model.predict_proba(data)[:, 2]]

def likelihood3(model,data):
    return [f"{prob*100:.2f}" for prob in model.predict_proba(data)[:, 3]]



def model_performance(data,predictions):
    cm = confusion_matrix(data['rul_class'], predictions)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    
    st.write()
    c1,c2,c3=st.columns(3)
    with c1:
       prec=precision_score(data['rul_class'], predictions,average='macro')
       st.write(f"#### Precision : {prec * 100:.2f}%")
    with c2:
       
       acc=accuracy_score(data['rul_class'], predictions)
       st.write(f"####  Accuracy : {acc*100:.2f}%")   
       
    with c3:
       
       F=f1_score(data['rul_class'], predictions,average='macro')
       st.write(f"#### F1 score : {F * 100:.2f} %")  
       
       
    st.write('\n\n')
    c1,c2=st.columns(2)
    with c1:
        st.write("#### True Positive Rate (TPR)")
        for i, tpr in enumerate(TPR):
           st.markdown(f"##### Class {i}<span style='padding-left: 20px; padding-right:20px;'>:</span> {tpr*100:.2f}%", unsafe_allow_html=True)
    with c2:
        st.write("#### False Positive Rate (FPR)")
        for i, fpr in enumerate(FPR):
           st.markdown(f"##### Class {i}<span style='padding-left: 20px; padding-right:20px;'>:</span> {fpr*100:.2f}%", unsafe_allow_html=True)


    
    st.write('\n\n')
    st.write("#### Confusion Matrix")
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    st.pyplot(plt)

   
    
       



    
    
    
