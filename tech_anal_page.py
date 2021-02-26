import streamlit as st
import data_loader as data 
import utils
import keras
import pandas as pd 
import workers as workers
import actions
import user_state as u 

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.layouts import gridplot

stock_data = data.get_stock_data()

def app(tar_stocks):
    pval_thres = st.sidebar.number_input("Significance Level", value = .05)
    days_to_look = st.sidebar.number_input("Number Of Days To Lookback", value = 7)
    u.tar_stocks = st.sidebar.multiselect(label = "Selected Stocks" , options = stock_data.columns)
    print(u.tar_stocks)
    col1, col2 = st.beta_columns((4,1)) 
    # Column 1
    with col1:
        with st.beta_container():
            st.title("Stock Analysis")
            ticker = st.selectbox(options = list(stock_data.columns), label = "Ticker")
            df_ticker = stock_data.filter([ticker]).dropna()
            st.line_chart(df_ticker)

        # AI - Prediction
        model = keras.models.load_model("lstm_models/f"+ ticker + "_lstm_model.h5")
        y_test_pred, y_test_real, dates = utils.test_LSTM_model(model, df_ticker, split_frac = 0.95, return_real_prices = True)
        pred_prices = utils.predict_prices(model,df_ticker)
        

        with st.beta_container():
            st.title("Model Performance")
            st.line_chart(pd.DataFrame({"Real": y_test_real.flatten(), "Predicted": y_test_pred.flatten()}))
            # st.bokeh_chart(workers.show_historical_prices_plot(y_test_real, y_test_pred, dates), use_container_width = True)
            # x = [1, 2, 3, 4, 5]
            # y = [6, 7, 2, 4, 5]
            # p = figure( title='simple line example', x_axis_label='x',y_axis_label='y')
            # p.line(x, y, legend_label='Trend', line_width=2)
            # st.bokeh_chart(p, use_container_width=True)

        with st.beta_container():
            st.title("Predicted Future Prices")
            st.line_chart(pred_prices[-7:])


    # Column 2
    with col2: 
        st.title("Historical Statistics")
        
        st.markdown("### Past " + str(days_to_look) + " Day Returns")
        st.markdown(workers.show_recent_returns(df_ticker, ticker, num_days = days_to_look))
        
        st.markdown("### Past " + str(days_to_look) + " Day Volatility")
        st.markdown(workers.show_recent_vols(df_ticker, ticker, num_days = days_to_look))
        
        st.markdown("### Variance Ratio")
        st.table(workers.show_vr_stats(df_ticker, ticker, [2,4,8,16,32], 2, pval_thres))
        
        st.markdown("### Has Technical Trading Opportunity")
        st.markdown(workers.show_if_random_walk(df_ticker, ticker, [2,4,8,16,32], 2, pval_thres))

        st.title("Projected Statistics")

        st.markdown("### Future Returns")
        st.markdown(workers.show_future_expected_returns(model, df_ticker, ticker))



