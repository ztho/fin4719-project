import streamlit as st
import data_loader as data 
from utils import *
import keras

stock_data = data.get_stock_data()

def app():
    col1, col2 = st.beta_columns([3,1])

    # Column 1
    col1.title("Stock Analysis")
    ticker = col1.selectbox(options = list(stock_data.columns), label = "Ticker")
    
    df_ticker = stock_data.filter([ticker]).dropna()
    col1.line_chart(df_ticker)
    
    vr_stats = get_all_vr_stats(df_ticker, ticker, k = 2, var_form = 2)
    is_random_walk = False if vr_stats[2] >= .05 else True
    recent_ret = get_recent_returns(df_ticker, ticker, num_days = 7)
    recent_vol = get_recent_vol(df_ticker, ticker, num_days = 7)

    # AI - Prediction
    model = keras.models.load_model("lstm_models/f"+ ticker + "_lstm_model.h5")
    
    y_test_pred, y_test_real = test_LSTM_model(model, df_ticker, split_frac = 0.95)
    col1.title("Model Prediction")
    col1.line_chart(pd.DataFrame({"Real": y_test_real.flatten(), "Predicted": y_test_pred.flatten()}))
    # Column 2
    col2.markdown("## Stock Statistics")
    col2.markdown("### Variance Ratio")
    col2.markdown(vr_stats[0])
    col2.markdown("### Has Technical Trading Opportunity")
    col2.markdown("Yes" if is_random_walk else "No")
    col2.markdown("### Past 7 Day Returns")
    col2.markdown(recent_ret)