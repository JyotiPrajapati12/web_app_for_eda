import streamlit as st
import functions

st.set_page_config(
    page_title="Exploratory Data Analysis of Training Data",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('Exploratory Data Analysis of Training Data')
data = functions.load_data('oil_cunsumption_training_data.csv')

functions.display_dataframe_info(data)

selected_function = st.sidebar.selectbox('Select Analysis Function', [
    'None',  
    'Target Column Analysis', 
    'Basic Statistics', 
    'Correlation Matrix', 
    'Numerical Data Analysis',
    'Box Plot',
    'Feature Importance'
])

output_placeholder = st.empty()

with output_placeholder.container():
    if selected_function == 'Numerical Data Analysis':
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        selected_column = st.sidebar.selectbox('Select Numerical Column', numerical_columns)
        if selected_column:
            st.subheader(f'Distribution Plot for {selected_column}')
            fig = functions.numerical_data_analysis(data, selected_column)
            st.plotly_chart(fig)
    elif selected_function == 'Box Plot':
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        selected_column = st.sidebar.selectbox('Select Numerical Column for Box Plot', numerical_columns)
        if selected_column:
            st.subheader(f'Box Plot for {selected_column}')
            fig = functions.box_plot(data, selected_column)
            st.plotly_chart(fig)
    elif selected_function == 'None':
        pass
    elif selected_function == 'Feature Importance':
        functions.feature_importance_train()
    else:
        eda_functions = {
            'Target Column Analysis': functions.target_column_analysis,
            'Basic Statistics': functions.basic_statistics,
            'Correlation Matrix': functions.correlation_matrix
        }
        eda_functions[selected_function](data)
