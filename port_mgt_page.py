import streamlit as st


def app():
    st.title("My First App")

    # Sidebar https://github.com/antonio-catalano/StockAnalysisApp/blob/master/app.py
    risk_pref = st.sidebar.number_input("Risk Preference")
    initial_cap = st.sidebar.number_input("Initial Capital")
    start_date = st.sidebar.date_input("Start Date")

    # Main Display 
    col1, col2 = st.beta_columns([3,1])

    col2.markdown("## Portfolio Statistics")