####
# Frontend script for technical analysis page
####

import streamlit as st
import pandas as pd 

# Import own modules
import data_loader as data 
import utils
import workers as workers
import actions
import user_state as u 

# import removed for static loading. See utils file. 
# import keras

stock_data = data.get_stock_data()

def app(tar_stocks):
    days_to_look = st.sidebar.number_input("Number Of Days To Lookback", value = 7)
    
    # For display purposes, no functional purpose
    st.sidebar.markdown("##### Technical Indicators")
    st.sidebar.checkbox("MACD", value = True)
    st.sidebar.checkbox("RSI", value = True)
    st.sidebar.checkbox("Arron Oscillator", value = False)
    st.sidebar.checkbox("Bollinger Bands", value = False)

    pval_thres = st.sidebar.number_input("Significance Level", value = .05)

    days_forward = st.sidebar.number_input("Number of Days to Predict", value = 21)

    u.tar_stocks = st.sidebar.multiselect(label = "Selected Stocks" , options = stock_data.columns)

    # days_lookback_pred = 90 (When using dynamic only. See Utils)

    # Top Component
    with st.beta_container():
        st.title("Stock Analysis")
        col1, col2 = st.beta_columns([5,2])

        # Left Column
        with col1:
            ticker = st.selectbox(options = list(stock_data.columns), label = "Ticker")
            df_ticker = stock_data.filter([ticker]).dropna()
            # historical chart
            st.markdown("### " + ticker + " Historical Price")
            st.bokeh_chart(workers.show_historical_prices_plot(df_ticker))
            # st.line_chart(df_ticker)

            # AI - Prediction - Dynamic. See Utils file
            # model = keras.models.load_model("lstm_models/f"+ ticker + "_lstm_model.h5")
            # y_test_pred, y_test_real, dates, mape = utils.test_LSTM_model(model, df_ticker, split_frac = 0.95, return_real_prices = True)
            # pred_prices = utils.predict_prices(model,df_ticker[-days_lookback_pred:], days_forward)

            # AI - Prediction - Static 
            y_test_pred, y_test_real, dates, mape = data.get_model_test_results(ticker)
            pred_prices = data.get_model_pred_fut_prices(ticker, stock_data)[:days_forward]

            st.markdown("### "+ ticker + " Model Performance")
            # model performance chart 
            st.bokeh_chart(workers.show_model_performance_plot(df_ticker, y_test_pred, y_test_real, 
                                                                dates, .95))
            st.markdown("Percentage Error " + str(round(mape, 3)) + "%")
    
        # Right Column 
        with col2:
            st.markdown("## Historical Statistics")
        
            st.markdown("#### Past " + str(days_to_look) + " Day Daily Returns")
            st.markdown(workers.show_recent_returns(df_ticker, ticker, num_days = days_to_look))
            
            st.markdown("#### Past " + str(days_to_look) + " Day Daily Volatility")
            st.markdown(workers.show_recent_vols(df_ticker, ticker, num_days = days_to_look))
            
            st.markdown("#### Variance Ratio")
            st.table(workers.show_vr_stats(df_ticker, ticker, [2,4,8,16,32], 2, pval_thres))
            
            st.markdown("#### Has Technical Trading Opportunity")
            st.markdown(workers.show_if_random_walk(df_ticker, ticker, [2,4,8,16,32], 2, pval_thres))

    # Bottom Component
    with st.beta_container():
        col3, col4 = st.beta_columns([5,2])
        # Left Column 
        with col3: 
            # st.line_chart(pd.DataFrame({"Real": y_test_real.flatten(), "Predicted": y_test_pred.flatten()}))
            st.markdown("### Predicted Future Prices")
            if days_forward > 0:
                st.bokeh_chart(workers.show_predicted_prices_plot(pred_prices, df_ticker, days_forward))
            else:
                st.markdown("No Graph To Display. Number of Days To Predict Must Be More Than 3")
        # Right Column
        with col4:
            st.markdown("## Projected Statistics")

            st.markdown("#### Future Returns " + str(workers.show_num_days_predicted(pred_prices, days_forward)))
            # st.markdown(workers.show_future_expected_returns(model, df_ticker, ticker, days_lookback_pred, 252, False, None ))
            st.markdown(workers.show_future_expected_returns(pred_prices, days_forward))
            st.markdown("#### Future Volatility " + str(workers.show_num_days_predicted(pred_prices, days_forward)))
            st.markdown(workers.show_future_expected_vols(pred_prices, days_forward))




    
        




