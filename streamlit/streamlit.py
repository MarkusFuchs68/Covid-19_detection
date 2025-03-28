import streamlit as st
st.title('Covid-19 Data Analysis')
st.write('This is a simple Streamlit app to analyze Covid-19 data.')
if st.checkbox('Display Data'):
    st.write('TODO:')
    # Assuming you have a DataFrame named df
    # st.dataframe(df)