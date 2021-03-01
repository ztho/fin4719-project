import streamlit as st 
import tech_anal_page as app1
import port_mgt_page as app2
import data_loader as data 
import utils
import workers
import user_state as u
from bokeh.plotting import figure
from PIL import Image

img = Image.open("img/icon.jpg")
st.set_page_config(page_title = "techinvestor.ai",
                   layout = "wide",
                   page_icon = img) # use whole page

def main():
    PAGES = {"Technical Analysis": app1, "Portfolio Management": app2}
    st.sidebar.title('Navigation')
    selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app(u)

if __name__ == "__main__":
    main()