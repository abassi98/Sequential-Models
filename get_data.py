#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:08:52 2021

@author: Alberto Bassi & Riccardo Tomada
"""

import websocket
import config
from   binance.client import Client
import pandas as pd
import numpy as np

# Get live data of a single crypto pair (BNB - USDT) using testnet (test)

client = Client(config.API_KEY, config.API_SECRET)

klines = np.array(client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_1HOUR, "7 Jun, 2017", "29 Oct, 2021"))

def binanceDataFrame(klines):
    df = pd.DataFrame(klines.reshape(-1,12), dtype=float, columns = ('Open Time',
                                                                     'Open',
                                                                     'High',
                                                                     'Low',
                                                                     'Close',
                                                                     'Volume',
                                                                     'Close time',
                                                                     'Quote asset volume',
                                                                     'Number of trades',
                                                                     'Taker buy base asset volume',
                                                                     'Taker buy quote asset volume',
                                                                     'Ignore'))

    df['Open Time']  = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    return df

df = binanceDataFrame(klines)
h  = df.to_csv('bitcoin_dataset')

# Note: Binance test mode reset every month. On test mode, only data of the current month are available.
    
# Binance constant to retrieve info of different time ranges (hourly, weekly,....):
    
# KLINE_INTERVAL_1MINUTE = '1m'
# KLINE_INTERVAL_3MINUTE = '3m'
# KLINE_INTERVAL_5MINUTE = '5m'
# KLINE_INTERVAL_15MINUTE = '15m'
# KLINE_INTERVAL_30MINUTE = '30m'
# KLINE_INTERVAL_1HOUR = '1h'
# KLINE_INTERVAL_2HOUR = '2h'
# KLINE_INTERVAL_4HOUR = '4h'
# KLINE_INTERVAL_6HOUR = '6h'
# KLINE_INTERVAL_8HOUR = '8h'
# KLINE_INTERVAL_12HOUR = '12h'
# KLINE_INTERVAL_1DAY = '1d'
# KLINE_INTERVAL_3DAY = '3d'
# KLINE_INTERVAL_1WEEK = '1w'
# KLINE_INTERVAL_1MONTH = '1M'

# open_time = np.array(df["Open Time"])
# high      = np.array(df["High"])
# low       = np.array(df['Low'])
# plt.plot(open_time, high, label = 'High')
# plt.plot(open_time, low, label = 'Low')
# plt.legend()
# plt.show()
