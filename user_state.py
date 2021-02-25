class User:
    def __init__(self):
        self.target_stocks = ['AAPL']
        self.risk_pref = None
    
    def get_target_stocks(self):
        return self.target_stocks

    def add_stocks_to_target(self, ticker):
        self.target_stocks = self.get_target_stocks().append(ticker) 
        return self.target_stocks
    
    