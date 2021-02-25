import streamlit as st
import data_loader as data 
import utils
import keras
import pandas as pd 
import workers as workers


stock_data = data.get_stock_data()

def app(User):
    pval_thres = st.sidebar.number_input("Significance Level", value = .05)
    days_to_look = st.sidebar.number_input("Number Of Days To Lookback", value = 7)
    add_to_list = st.sidebar.checkbox("Add To List", value = False)
    col1, col2 = st.beta_columns((4,1)) 
    print(User.get_target_stocks())
    # Column 1
    with col1:
        with st.beta_container():
            st.title("Stock Analysis")
            ticker = st.selectbox(options = list(stock_data.columns), label = "Ticker")
            df_ticker = stock_data.filter([ticker]).dropna()
            st.line_chart(df_ticker)
        
        # vr_stats = utils.test_multiple_periods(df_ticker, ticker,)
        # is_random_walk = False if all(p > .1 for p in vr_stats['pval'].values) else True

        # AI - Prediction
        model = keras.models.load_model("lstm_models/f"+ ticker + "_lstm_model.h5")
        y_test_pred, y_test_real = utils.test_LSTM_model(model, df_ticker, split_frac = 0.95)
        
        with st.beta_container():
            st.title("Model Performance")
            st.line_chart(pd.DataFrame({"Real": y_test_real.flatten(), "Predicted": y_test_pred.flatten()}))

        with st.beta_container():
            st.title("Predicted Future Prices")

    # Column 2
    with col2: 
        st.title("Stock Statistics")
        
        st.markdown("### Past " + str(days_to_look) + " Day Returns")
        st.markdown(workers.show_recent_returns(df_ticker, ticker, num_days = days_to_look))
        
        st.markdown("### Past " + str(days_to_look) + " Day Volatility")
        st.markdown(workers.show_recent_vols(df_ticker, ticker, num_days = days_to_look))
        
        st.markdown("### Variance Ratio")
        st.table(workers.show_vr_stats(df_ticker, ticker, [2,4,8,16,32], 2))
        
        st.markdown("### Has Technical Trading Opportunity")
        st.markdown(workers.show_if_random_walk(df_ticker, ticker, [2,4,8,16,32], 2, pval_thres))
    
    if add_to_list and ticker not in User.get_target_stocks(): 
        User.add_stocks_to_target(ticker)

