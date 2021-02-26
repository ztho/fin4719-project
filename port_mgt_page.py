import streamlit as st
import data_loader as data 
import user_state as u 
import workers 

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.models import NumeralTickFormatter, HoverTool
import numpy as np

stock_data = data.get_stock_data()
rel_perf_data = data.get_rel_perf()

def app(tar_stocks):

    # Sidebar https://github.com/antonio-catalano/StockAnalysisApp/blob/master/app.py
    days_sim = st.sidebar.number_input("Number of Simulation Days", value = 30)
    init_cap = st.sidebar.number_input("Initial Capital", value = 1000)
    rf = st.sidebar.number_input("Risk Free Rate", value  = .0164)
    u.tar_stocks = st.sidebar.multiselect(label = "Selected Stocks" , options = stock_data.columns)
    
    
    opt_weights = workers.exec_get_black_litterman_optimal_weights(u.tar_stocks, stock_data, rel_perf_data)
    # st.write(opt_weights)
    sim = workers.exec_run_cppi(opt_weights, stock_data,u.tar_stocks)
    
    # Main Display 
    with st.beta_container():
        col1, col2 = st.beta_columns([4, 1])
        with col1:
            if sim is not None:
                p = figure(x_axis_type='datetime', title='Portfolio Historical Performance')
                p.grid.grid_line_alpha = 0
                p.xaxis.axis_label = 'Date'
                p.yaxis.axis_label = 'Price'
                p.ygrid.band_fill_color = 'cornflowerblue'
                p.ygrid.band_fill_alpha = 0.1
                p.yaxis.formatter = NumeralTickFormatter(format="$0,0") 
                p.line(np.array(sim.index, dtype=np.datetime64), sim['Portfolio Value'], color='darkslateblue')
                
                p.add_tools(HoverTool(tooltips=[('Date', '$x{%F}'),
                                                ('Price', '$$y{0,0.00}')],
                                    formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                                    mode='vline')) #display a tooltip whenever the cursor is vertically in line with a glyph
                st.bokeh_chart(p)
            df = workers.show_optimal_weights(opt_weights, u.tar_stocks)
            if df is not None:
                st.table(workers.show_optimal_weights(opt_weights, u.tar_stocks))
            else:
                st.write("No Allocation Feasible")

        with col2:
            st.title("Portfolio Statistcs")
            
            st.markdown("#### Expected Returns")
            st.markdown(workers.show_portfolio_returns(opt_weights, stock_data, u.tar_stocks))
            
            st.markdown("#### Expected Volatility")
            st.markdown(workers.show_portfolio_vols(opt_weights, stock_data, u.tar_stocks))
            
            st.markdown("#### Value at Risk")
            st.markdown(workers.show_VaR(sim, days_sim, init_cap, rf))       


