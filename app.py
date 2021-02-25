import streamlit as st 
import port_mgt_page as app1
import tech_anal_page as app2
import numpy as np 
import pandas as pd 
from user_state import User
import data_loader as data 
import utils
import workers

st.set_page_config(layout="wide")
USER = User()
etf_data = data.get_etf_data()

PAGES = {"Portfolio Management": app1, "Technical Analysis": app2}
st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))


page = PAGES[selection]
page.app(USER)