####
# Module to interact with user state
###

import user_state as u

def add_stock(u, cur_ticker):
    if cur_ticker not in u.tar_stocks:
        u.tar_stocks.append(cur_ticker)

def remove_stock(u, cur_ticker):
    if cur_ticker in u.tar_stocks:
        u.tar_stocks.remove(cur_ticker)