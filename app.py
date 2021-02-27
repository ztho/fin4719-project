import streamlit as st 
import tech_anal_page as app1
import port_mgt_page as app2
import numpy as np 
import pandas as pd 
import data_loader as data 
import utils
import workers
import user_state as u
from bokeh.plotting import figure

# try:
#     # Before Streamlit 0.65
#     from streamlit.ReportThread import get_report_ctx
#     from streamlit.server.Server import Server
# except ModuleNotFoundError:
#     # After Streamlit 0.65
#     from streamlit.report_thread import get_report_ctx
#     from streamlit.server.server import Server


st.set_page_config(layout="wide")

def main():
    PAGES = {"Technical Analysis": app1, "Portfolio Management": app2}
    st.sidebar.title('Navigation')
    selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app(u.tar_stocks)



if __name__ == "__main__":
    main()