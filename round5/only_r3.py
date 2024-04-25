import json
import math
import pandas as pd
import numpy as np
import copy
import collections
from collections import defaultdict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import pandas as pd

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 
              'ORCHIDS' : 0, 
              'GIFT_BASKET' : 0, 'STRAWBERRIES' : 0, 'CHOCOLATE' : 0, 'ROSES' : 0,
              'COCONUT' : 0, 'COCONUT_COUPON' : 0}

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
                        'GIFT_BASKET' : 60, 'STRAWBERRIES' : 350, 'CHOCOLATE' : 250, 'ROSES' : 60,
                        'COCONUT' : 300, 'COCONUT_COUPON' : 600}

    
    def __init__(self):
        self.prices = {"PRODUCT1": [9], 
                       "PRODUCT2": [150],
                       "AMETHYSTS" : [10000],
                       "STARFRUIT" : [],
                       "ORCHIDS" : [],
                       "CHOCOLATE" : [],
                       "STRAWBERRIES" : [],
                       "TOTAL_VOL_ORCHIDS" : [],
                       "SPREAD" : [],
                       "BASKET_COMPONENTS" : [],
                       "COCONUT" : [],
                       "COCONUT_COUPON" : []}  # Initialize prices for each symbol
        
        self.spreads = {"GIFT_BASKET" : [],
                        "STRAW_CHOCO" : [],
                        "CHOCO_ROSE" : [],
                        "ROSE_STRAW" : [],
                        "CHOCO_BASKET" : [],
                        "ROSE_BASKET" : [],
                        "BASKET_COMPONENTS" : [],
                        "ROSE_CHOC" : []}
        
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
                                  "ROSES" : 60,
                                  "COCONUT" : 300,
                                  "COCONUT_COUPON" : 600}  # Maximum positions for each symbol
                                  
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
        roll, window = False, 1000               ###
        closing, close_at = False, -0.5         ###
        std_coeff = 0.5                         ###
                                                ###
        ###########################################


        if roll and len(self.spreads['BASKET_COMPONENTS']) > window:
            rolling_spread = self.spreads['BASKET_COMPONENTS'][-window:]
            ma_spread = np.mean(rolling_spread)
            basket_std = np.std(rolling_spread)
        else:
            basket_std = 76.424
            ma_spread = 379

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



        return orders
    

    def compute_orders_ma_choc(self, state):

        positions = state.position
        order_depth = state.order_depths

        orders = {'CHOCOLATE' : []}
        prods = ['CHOCOLATE']

        self.position['CHOCOLATE'] = positions.get('CHOCOLATE', 0)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}


        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2

        

        product = 'CHOCOLATE'
        person = 'Vinnie'
        coeff = 0.15

        rem_buys = self.POSITION_LIMIT[product] - self.position[product]
        rem_sells = self.position[product] + self.POSITION_LIMIT[product]

        buys = 0
        sells = 0

        if product in state.market_trades:
            market_trades = state.market_trades[product] # list of trades for chocolate
            for trade in market_trades:
                if trade.buyer == person:
                    buys += coeff * trade.quantity

        person = 'Remy'
        coeff = 0.15

        if product in state.market_trades:
            market_trades = state.market_trades[product] # list of trades for chocolate
            for trade in market_trades:
                if trade.seller == person:
                    sells += coeff * trade.quantity

        trade = buys - sells

        if trade > 0 and rem_buys > 0:
            size = math.floor(min(trade, rem_buys))
            orders[product].append(Order(product, worst_sell[product], size))
        
        if trade < 0 and rem_sells > 0:
            size = math.floor(min(-trade, rem_sells))
            orders[product].append(Order(product, worst_buy[product], -size))


        return orders
            
    
    def compute_orders_ma_straw(self, order_depth, positions):

        orders = {'STRAWBERRIES' : []}
        self.position['STRAWBERRIES'] = positions.get('STRAWBERRIES', 0)

        rem_buys, rem_sells = {}, {}
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}


        for p in ['STRAWBERRIES']:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            self.tot_vol[p] += abs(self.position[p] - self.last_pos[p])
            mid_price[p] = (best_sell[p] + best_buy[p])/2

            rem_buys[p] = self.POSITION_LIMIT[p] - self.position[p]
            rem_sells[p] = self.position[p] + self.POSITION_LIMIT[p]

        self.prices['STRAWBERRIES'].append(mid_price['STRAWBERRIES'])

        window_size = 500
        pos_div = 240
        std_coeff = 2  # Number of standard deviations for Bollinger Bands
        rsi_period = 14  # Period for calculating Relative Strength Index (RSI)
        rsi_threshold = 30  # Threshold for considering oversold conditions

        choco_sells = 0
        choco_buys = 0


        # Calculate Bollinger Bands
        if len(self.prices['STRAWBERRIES']) > window_size + 1:
            rolling_spread = self.prices['STRAWBERRIES'][-window_size:]
            rolling_mean = np.mean(rolling_spread)
            rolling_std = np.std(rolling_spread)

            upper_band = rolling_mean + std_coeff * rolling_std
            lower_band = rolling_mean - std_coeff * rolling_std

            # Calculate RSI
            price_changes = np.diff(self.prices['STRAWBERRIES'][-rsi_period:])
            gains = price_changes[price_changes >= 0]
            losses = -price_changes[price_changes < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Check if the current price is above the upper band or below the lower band
            current_price = self.prices['STRAWBERRIES'][-1]
            if current_price > upper_band:  # Price is above the upper band, sell chocolate
                size_choco = min(round(self.POSITION_LIMIT['STRAWBERRIES'] / pos_div), rem_sells['STRAWBERRIES'])
                choco_sells += size_choco
                rem_sells['STRAWBERRIES'] -= size_choco
            elif current_price < lower_band and rsi < rsi_threshold:  # Price is below the lower band and RSI indicates oversold conditions, buy chocolate (short position)
                size_choco = min(round(self.POSITION_LIMIT['STRAWBERRIES'] / pos_div), rem_buys['STRAWBERRIES'])
                choco_buys += size_choco
                rem_buys['STRAWBERRIES'] -= size_choco

        if choco_buys > 0:
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], choco_buys))
        if choco_sells > 0:
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -choco_sells))
        
        return orders


    def compute_orders_roses_choc(self, state):

        positions = state.position
        order_depth = state.order_depths

        orders = {'ROSES' : []}
        prods = ['ROSES']

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


        product = 'ROSES'
        person = 'Rhianna'
        coeff = 20

        rem_buys = self.POSITION_LIMIT[product] - self.position[product]
        rem_sells = self.position[product] + self.POSITION_LIMIT[product]

        buys = 0
        sells = 0

        if product in state.market_trades:
            market_trades = state.market_trades[product] # list of trades for roses
            for trade in market_trades:
                if trade.buyer == person:
                    buys += trade.quantity
                if trade.seller == person:
                    sells += trade.quantity

        trade = buys - sells

        if trade > 0 and rem_buys > 0:
            size = coeff * min(trade, rem_buys)
            orders[product].append(Order(product, worst_sell[product], size))
        
        if trade < 0 and rem_sells > 0:
            size = coeff * min(-trade, rem_sells)
            orders[product].append(Order(product, worst_buy[product], -size))


        return orders    
        


    def compute_orders_coconuts(self, state):

        positions = state.position
        order_depth = state.order_depths

        orders = {'COCONUT' : []}
        prods = ['COCONUT']

        self.position['COCONUT'] = positions.get('COCONUT', 0)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2


        product = 'COCONUT'
        person = 'Rhianna'
        coeff = 5

        rem_buys = self.POSITION_LIMIT[product] - self.position[product]
        rem_sells = self.position[product] + self.POSITION_LIMIT[product]

        buys = 0
        sells = 0

        if product in state.market_trades:
            market_trades = state.market_trades[product] # list of trades for roses
            for trade in market_trades:
                if trade.buyer == person:
                    sells += trade.quantity
                if trade.seller == person:
                    buys += trade.quantity

        trade = buys - sells

        if trade > 0 and rem_buys > 0:
            size = coeff * min(trade, rem_buys)
            orders[product].append(Order(product, worst_sell[product], size))
        
        if trade < 0 and rem_sells > 0:
            size = coeff * min(-trade, rem_sells)
            orders[product].append(Order(product, worst_buy[product], -size))


        return orders       
    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        result = {}
        
        # ords = self.compute_orders_basket(state.order_depths, state.position)
        # result['GIFT_BASKET'] = ords['GIFT_BASKET']

        # ords = self.compute_orders_roses_choc(state)
        # result['ROSES'] = ords['ROSES']

        # ords = self.compute_orders_ma_choc(state)
        # result['CHOCOLATE'] = ords['CHOCOLATE']

        # ords = self.compute_orders_ma_straw(state.order_depths, state.position)
        # result['STRAWBERRIES'] = ords['STRAWBERRIES']     
            
        ords = self.compute_orders_coconuts(state)
        result['COCONUT'] = ords['COCONUT']


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data






