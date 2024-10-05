import streamlit as st
import functions
import plotly.express as px  


st.set_page_config(
    page_title="Model Testing",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("Model Testing")

data = None

uploaded_file = st.file_uploader("Choose a CSV file for testing", type="csv")


if uploaded_file is not None:
    
    data = functions.load_data(uploaded_file)
    st.dataframe(data.head(10))
    #functions.display_dataframe_info(data)
    
    st.header('Model Prediction')
    
    model = functions.load_model('model.pkl')

    if model is not None:
        data1=data.drop(['rul_class'],axis=1)
        predictions = functions.predict(model, data1)
        likelihood0=functions.likelihood0(model,data1)
        likelihood1=functions.likelihood1(model,data1)
        likelihood2=functions.likelihood2(model,data1)
        likelihood3=functions.likelihood3(model,data1)
        data1['Predictions'] = predictions
        data1['Likelihood of class 0']=likelihood0
        data1['Likelihood of class 1']=likelihood1
        data1['Likelihood of class 2']=likelihood2
        data1['Likelihood of class 3']=likelihood3
        st.subheader('Data with Predictions:')
        st.dataframe(data1)
    

    
       
    else:
        st.error("Failed to load the model. Please ensure the correct scikit-learn version is installed.")



selected_function = st.sidebar.selectbox('Select Analysis Function', [
    'None',  
    'Model Performance',
    'Target Column Analysis', 
    'Basic Statistics', 
    'Correlation Matrix', 
    'Numerical Data Analysis',
    'Box Plot',
    'Feature Importance'
])


output_placeholder = st.empty()

with output_placeholder.container():
    if data is not None:
        if selected_function == 'Numerical Data Analysis':
            numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
            if not numerical_columns.empty:
                selected_column = st.sidebar.selectbox('Select Numerical Column', numerical_columns)
                if selected_column:
                    st.subheader(f'Distribution Plot for {selected_column}')
                    fig = functions.numerical_data_analysis(data, selected_column)
                    st.plotly_chart(fig)
        elif selected_function == 'Box Plot':
            numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
            if not numerical_columns.empty:
                selected_column = st.sidebar.selectbox('Select Numerical Column for Box Plot', numerical_columns)
                if selected_column:
                    st.subheader(f'Box Plot for {selected_column}')
                    fig = functions.box_plot(data, selected_column)
                    st.plotly_chart(fig)
        elif selected_function=='Model Performance':
            st.write('\n\n')
            st.header(' Model Performance')
            functions.model_performance(data,data1['Predictions'])
            
                    
        
            
        elif selected_function == 'Feature Importance':
            functions.feature_importance_train()    
        elif selected_function == 'None':
            pass
        else:
            st.header("Exploratory Data Analysis of Testing Data")
            
            eda_functions = {
                'Target Column Analysis': functions.target_column_analysis,
                'Basic Statistics': functions.basic_statistics,
                'Correlation Matrix': functions.correlation_matrix
            }
            
            if selected_function in eda_functions:
                eda_functions[selected_function](data)
    else:
        st.warning("Please upload a CSV file to proceed with the analysis.")
