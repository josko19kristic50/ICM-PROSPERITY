import json
import math
import pandas as pd
import copy
import collections
from collections import defaultdict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 
              'ORCHIDS' : 0, 
              'GIFT_BASKET' : 0, 'STRAWBERRIES' : 0, 'CHOCOLATE' : 0, 'ROSES' : 0}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 
                        'ORCHIDS' : 100, 
                        'GIFT_BASKET' : 60, 'STRAWBERRIES' : 350, 'CHOCOLATE' : 250, 'ROSES' : 60}

    
    def __init__(self):
        self.prices = {"PRODUCT1": [9], 
                       "PRODUCT2": [150],
                       "AMETHYSTS" : [10000],
                       "STARFRUIT" : [],
                       "ORCHIDS" : [],
                       "TOTAL_VOL_ORCHIDS" : [],
                       "SPREAD" : []}  # Initialize prices for each symbol
        
        self.spreads = {"ROSE_CHOC" : [],
                        "BASKET_COMPONENTS" : [],
                        "CHOC_STRAW" : []}
        
        self.last_pos = {'STRAWBERRIES' : 0, 'CHOCOLATE' : 0, 'ROSES' : 0, 'GIFT_BASKET' : 0}
        self.tot_vol = {'STRAWBERRIES' : 0, 'CHOCOLATE' : 0, 'ROSES' : 0, 'GIFT_BASKET' : 0}


        self.maximum_positions = {"PRODUCT1": 20, 
                                  "PRODUCT2": 20,
                                  "AMETHYSTS": 20,
                                  "STARFRUIT": 20,
                                  "ORCHIDS" : 100,
                                  "GIFT_BASKET" : 60,
                                  "STRAWBERRIES" : 350,
                                  "CHOCOLATE" : 250,
                                  "ROSES" : 60}  # Maximum positions for each symbol

        self.ratios = { "STRAWBERRIES" : [],
                       "CHOCOLATE" : [],
                       "ROSES" : []
        }
                                  
        self.bids = {"PRODUCT1": [], 
                          "PRODUCT2": [],
                          "AMETHYSTS": [],
                          "STARFRUIT": [],
                          "ORCHIDS": []      ,                            
                          "GIFT_BASKET" : [],
                            "STRAWBERRIES" : [],
                            "CHOCOLATE" : [],
                            "ROSES" : []}  # Bids history
        
        self.asks = {"PRODUCT1": [], 
                          "PRODUCT2": [],
                          "AMETHYSTS": [],
                          "STARFRUIT": [],
                          "ORCHIDS": []      ,                            
                          "GIFT_BASKET" : [],
                            "STRAWBERRIES" : [],
                            "CHOCOLATE" : [],
                            "ROSES" : []} # Asks history
                          
        self.TOTAL_VOL_ORCHIDS = 0
        self.LAST_POSITION = 0
                         
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
        


    def compute_orders_roses_choc(self, order_depth, positions):

        orders = {'CHOCOLATE' : [], 'ROSES' : []}
        prods = ['CHOCOLATE', 'ROSES']

        self.position['CHOCOLATE'] = positions.get('CHOCOLATE', 0)
        self.position['ROSES'] = positions.get('ROSES', 0)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2

        coef = 15/8
        self.spreads["ROSE_CHOC"].append(mid_price['ROSES'] - coef * mid_price['CHOCOLATE'])

        #### HYPERPARAMS ##########################
                                                ###
        roll, window = True, 2000               ###
        closing, close_at = True, 0             ###
        trade_at = 1                            ###
                                                ###
        ###########################################


        if roll and len(self.spreads['ROSE_CHOC']) > window:
            rolling_spread = self.spreads['ROSE_CHOC'][-window:]
            mean = pd.Series(rolling_spread).mean()
            std = pd.Series(rolling_spread).std()
        else:
            mean = -344
            std = 105.16

        res = self.spreads['ROSE_CHOC'][-1]
        z_score = (res - mean)/std

        if z_score > trade_at:
            vol = self.position['ROSES'] + self.POSITION_LIMIT['ROSES']
            assert(vol >= 0)
            if vol > 0:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol))
        elif z_score < -trade_at:
            vol = self.POSITION_LIMIT['ROSES'] - self.position['ROSES']
            assert(vol >= 0)
            if vol > 0:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))
        elif closing and z_score < close_at and self.position['ROSES'] < 0:
            vol = -self.position['ROSES']
            assert(vol >= 0)
            if vol > 0:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))
        elif closing and z_score > -close_at and self.position['ROSES'] > 0:
            vol = self.position['ROSES']
            assert(vol >= 0)
            if vol > 0:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol))

        return orders    
    
    
    def compute_orders_basket(self, order_depth, positions):


        self.position['GIFT_BASKET'] = positions.get('GIFT_BASKET', 0)
        self.position['STRAWBERRIES'] = positions.get('STRAWBERRIES', 0)
        self.position['CHOCOLATE'] = positions.get('CHOCOLATE', 0)
        self.position['ROSES'] = positions.get('ROSES', 0)

        rem_buy = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET'] # Pozitivan broj
        rem_sell = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET'] # Pozitivan broj

        orders = {'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}

        for p in prods: # Get prices and volumes
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
                
        spread = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES']
        self.prices['SPREAD'].append(spread)
        self.spreads['BASKET_COMPONENTS'].append(spread)


        #### HYPERPARAMS ##########################
                                                ###
        roll, window = False, 500               ###
        closing, close_at = False, -0.5             ###
        std_coeff = 0.5                            ###
                                                ###
        ###########################################


        if roll and len(self.spreads['BASKET_COMPONENTS']) > window:
            rolling_spread = self.spreads['BASKET_COMPONENTS'][-window:]
            ma_spread = pd.Series(rolling_spread).mean()
            basket_std = pd.Series(rolling_spread).std()
        else:
            basket_std = 76.424
            ma_spread = 355

        z_score = (spread - ma_spread)/basket_std


        if z_score > std_coeff:
            vol = rem_sell
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))
        elif z_score < - std_coeff:
            vol = rem_buy
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
        elif closing and z_score < close_at and self.position['GIFT_BASKET'] < 0:
            vol = -self.position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
        elif closing and z_score > -close_at and self.position['GIFT_BASKET'] > 0:
            vol = self.position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))




        return orders

        
    def compute_orders_choc_straw(self, order_depth, positions):

        orders = {'ROSES' : [], 'STRAWBERRIES' : []}
        prods = ['ROSES', 'STRAWBERRIES']

        self.position['ROSES'] = positions.get('ROSES', 0)
        self.position['STRAWBERRIES'] = positions.get('STRAWBERRIES', 0)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2

        coef = 4/15
        self.spreads["CHOC_STRAW"].append(mid_price['STRAWBERRIES'] - coef * mid_price['ROSES'])

        #### HYPERPARAMS ##########################
                                                ###
        roll, window = True, 1000               ###
        closing, close_at = False, 0             ###
        trade_at = 1                            ###
                                                ###
        ###########################################


        if roll and len(self.spreads['CHOC_STRAW']) > window:
            rolling_spread = self.spreads['CHOC_STRAW'][-window:]
            mean = pd.Series(rolling_spread).mean()
            std = pd.Series(rolling_spread).std()
        else:
            mean = self.spreads['CHOC_STRAW'][-1]
            std = 9999

        res = self.spreads['CHOC_STRAW'][-1]
        z_score = (res - mean)/std

        if z_score > trade_at:
            vol = self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES']
            # s_vol = self.POSITION_LIMIT['STRAWBERRIES'] - self.position['STRAWBERRIES']
            assert(vol >= 0)
            if vol > 0:
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol))
                # orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], s_vol))
        elif z_score < -trade_at:
            vol = self.POSITION_LIMIT['STRAWBERRIES'] - self.position['STRAWBERRIES']
            # s_vol = self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES']
            assert(vol >= 0)
            if vol > 0:
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))
                # orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], -s_vol))
        elif closing and z_score < close_at and self.position['STRAWBERRIES'] < 0:
            vol = -self.position['CHOCOLATE']
            assert(vol >= 0)
            if vol > 0:
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))
        elif closing and z_score > -close_at and self.position['STRAWBERRIES'] > 0:
            vol = self.position['STRAWBERRIES']
            assert(vol >= 0)
            if vol > 0:
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol))

        return orders    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        
        timestamp = state.timestamp

        result = {}
        
        ords = self.compute_orders_basket(state.order_depths, state.position)
        result['GIFT_BASKET'] = ords['GIFT_BASKET']

        ords = self.compute_orders_roses_choc(state.order_depths, state.position)
        result['ROSES'] = ords['ROSES']

        ords = self.compute_orders_choc_straw(state.order_depths, state.position)
        # result['CHOCOLATE'] = ords['CHOCOLATE']
        result['STRAWBERRIES'] = ords['STRAWBERRIES']


        

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data






