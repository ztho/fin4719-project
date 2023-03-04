# techinvestor.ai
techinvestor.ai is a project used in FIN4719 FinTech and Financial Data Analytics module.
Project is available at: https://technical-ai-dashboard.onrender.com/

## About The Project
This project is a proof-of-concept, predicting future price movements through a Long Short-Term Memory (LSTM) model, which is used to generate a quantitative view on future stock performances. These forward projections are then fed into the Black-Litterman algorithm, which traditionally relied on subjective portfolio manager views for portfolio allocation. Project is created using Streamlit, graphs created from Bokeh. 

### Ideology
The screen first checks for presence of non-random walk behaviour as described in [(Lo and Mackinlay,1988)](https://www.jstor.org/stable/2962126) which could provide a technical trading opportunity. An LSTM model is then run to provide the future predicted prices, based on favoured technical indicators e.g MACD, RSI and historical price. Stocks can then be selected based on the future predicted price. We then use the Black-Litterman model to construct an optimal portfolio. The portfolio management tab runs a simulation and VaR analysis on the portfolio. 

### Other Notes
* Data used in this project is static 
* Due to constraints, the number of stocks available for selection is limited 
* Models are pre-trained due to limited resources
