import json
import math
import pandas as pd
import numpy as np
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
                       "CHOCOLATE" : [],
                       "STRAWBERRIES" : [],
                       "TOTAL_VOL_ORCHIDS" : [],
                       "SPREAD" : [],
                       "BASKET_COMPONENTS" : []}  # Initialize prices for each symbol
        
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
                                  "ROSES" : 60}  # Maximum positions for each symbol
                                  
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
        
    
    def make_prediction(self, bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
                         bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
                         mid_price_2_rows_earlier, mid_price_3_rows_earlier,
                         mid_price_4_rows_earlier, bid_price_1_row_before,
                         ask_price_1_row_before):

        # Coefficients
        coefficients = {
            'const': -1.4308,
            'bid_price_1': -0.0346,
            'bid_volume_1': 0.0087,
            'ask_price_1': -0.1000,
            'ask_volume_1': -0.0078,
            'bid_price_2': 0.4801,
            'bid_volume_2': 0.0148,
            'ask_price_2': 0.5418,
            'ask_volume_2': -0.0185,
            'mid_price_2_rows_earlier': 0.0268,
            'mid_price_3_rows_earlier': 0.0163,
            'mid_price_4_rows_earlier': 0.0316,
            'bid_price_1_row_before': 0.0146,
            'ask_price_1_row_before': 0.0237
        }

        # Initialize prediction
        prediction = coefficients['const']

        # Add contribution from each variable
        for variable in coefficients.keys():
            if variable != 'const':
                prediction += coefficients[variable] * locals()[variable]

        return round(prediction)
        
        
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
    

    def compute_orders_ma_choc(self, order_depth, positions):

        orders = {'CHOCOLATE' : []}
        self.position['CHOCOLATE'] = positions.get('CHOCOLATE', 0)

        rem_buys, rem_sells = {}, {}
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}


        for p in ['CHOCOLATE']:
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

        self.prices['CHOCOLATE'].append(mid_price['CHOCOLATE'])

        window_size = 500
        pos_div = 240
        std_coeff = 2  # Number of standard deviations for Bollinger Bands
        rsi_period = 14  # Period for calculating Relative Strength Index (RSI)
        rsi_threshold = 30  # Threshold for considering oversold conditions

        choco_sells = 0
        choco_buys = 0

        # Calculate Bollinger Bands
        if len(self.prices['CHOCOLATE']) > window_size + 1:
            rolling_spread = self.prices['CHOCOLATE'][-window_size:]
            rolling_mean = np.mean(rolling_spread)
            rolling_std = np.std(rolling_spread)

            upper_band = rolling_mean + std_coeff * rolling_std
            lower_band = rolling_mean - std_coeff * rolling_std

            # Calculate RSI
            price_changes = np.diff(self.prices['CHOCOLATE'][-rsi_period:])
            gains = price_changes[price_changes >= 0]
            losses = -price_changes[price_changes < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Check if the current price is above the upper band or below the lower band
            current_price = self.prices['CHOCOLATE'][-1]
            if current_price > upper_band:  # Price is above the upper band, sell chocolate
                size_choco = min(round(self.POSITION_LIMIT['CHOCOLATE'] / pos_div), rem_sells['CHOCOLATE'])
                choco_sells += size_choco
                rem_sells['CHOCOLATE'] -= size_choco
            elif current_price < lower_band and rsi < rsi_threshold:  # Price is below the lower band and RSI indicates oversold conditions, buy chocolate (short position)
                size_choco = min(round(self.POSITION_LIMIT['CHOCOLATE'] / pos_div), rem_buys['CHOCOLATE'])
                choco_buys += size_choco
                rem_buys['CHOCOLATE'] -= size_choco

        if choco_buys > 0:
            orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], choco_buys))
        if choco_sells > 0:
            orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -choco_sells))

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

        ords = self.compute_orders_ma_choc(state.order_depths, state.position)
        result['CHOCOLATE'] = ords['CHOCOLATE']

        ords = self.compute_orders_ma_straw(state.order_depths, state.position)
        result['STRAWBERRIES'] = ords['STRAWBERRIES']

        for product in ["ORCHIDS"]:
            product_orders = []            
            
            position = state.position.get(product, 0)            

            conversion_bid = state.observations.conversionObservations[product].bidPrice
            conversion_ask = state.observations.conversionObservations[product].askPrice
            import_tariff = state.observations.conversionObservations[product].importTariff
            export_tariff = state.observations.conversionObservations[product].exportTariff
            transport_fees = state.observations.conversionObservations[product].transportFees

            
            
            
            true_south_ask = conversion_ask + import_tariff + transport_fees
            true_south_bid = conversion_bid - export_tariff - transport_fees

            conversions = -position
            qty = 100

            product_orders.append(Order(product, math.floor(true_south_ask+1), -qty))
            product_orders.append(Order(product, math.ceil(true_south_bid-1), qty))
                
                
                
            print(" &&&&&&& TOTAL ORCHIDS VOL SO FAR:" , self.TOTAL_VOL_ORCHIDS, "&&&&&&&&&")
            result[product] = product_orders
            
            
            
            
        ######################################################################################


        for product in ["AMETHYSTS", "STARFRUIT"]:
            order_depth = state.order_depths[product]
            
            sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
            
            product_orders = []
            
            sell_vol, best_sell_pr = self.values_extract(sell_orders)
            buy_vol, best_buy_pr = self.values_extract(buy_orders, 1)
            
            # Define positions and maximum positions for the current symbol
            position = state.position.get(product, 0)
            max_position = self.maximum_positions.get(product, 0)
            new_position = position
            
            rem_buy = max_position - position
            rem_sell = position + max_position
            

            print("####### TRADING:  ", product, "###########")
            print(order_depth.sell_orders)
            print(order_depth.buy_orders, "\n")
            print("Highest bid: ", best_buy_pr)
            print("Lowest ask: ", best_sell_pr)
            print("  Position:   ", position, "   ")

            
            ########### UPDATE HISTORY
            

            avg_price_update = (best_sell_pr + best_buy_pr) / 2
            bid_update = best_buy_pr
            ask_update = best_sell_pr
            
            if len(self.prices[product]) > 0:
                # Append values to the lists
                self.prices[product].append(avg_price_update)
                self.bids[product].append(bid_update)
                self.asks[product].append(ask_update)

                # Remove the first element of each list
                self.prices[product] = self.prices[product][1:]
                self.bids[product] = self.bids[product][1:]
                self.asks[product] = self.asks[product][1:]
            else:
                self.prices[product].extend([avg_price_update, avg_price_update, avg_price_update, avg_price_update, avg_price_update])
                self.bids[product].extend([bid_update, bid_update])
                self.asks[product].extend([ask_update, bid_update])
            
            ###########################################################
            ####### STARFRUIT #########################################       
            ###########################################################
           
            
            if product == "STARFRUIT":
            
                new_position = position
                
                mid_price_1 = self.prices[product][-1]
                mid_price_2 = self.prices[product][-2]
                mid_price_3 = self.prices[product][-3]
                mid_price_4 = self.prices[product][-4]
                mid_price_5 = self.prices[product][-5]
                
                # Make copies of the dictionaries manually
                sell_orders_copy = {key: value for key, value in sell_orders.items()}
                buy_orders_copy = {key: value for key, value in buy_orders.items()}

                # Ensure there are at least two items in sell_orders_copy
                if len(sell_orders_copy) < 2:
                    first_sell_key, first_sell_value = next(iter(sell_orders_copy.items()))
                    sell_orders_copy[first_sell_key + 1] = 0  # Add a second item with volume 0

                # Ensure there are at least two items in buy_orders_copy
                if len(buy_orders_copy) < 2:
                    first_buy_key, first_buy_value = next(iter(buy_orders_copy.items()))
                    buy_orders_copy[first_buy_key - 1] = 0  # Add a second item with volume 0

                # Now you can safely access the first and second items
                ask_price_1, ask_volume_1 = list(sell_orders_copy.items())[0]
                ask_price_2, ask_volume_2 = list(sell_orders_copy.items())[1]

                bid_price_1, bid_volume_1 = list(buy_orders_copy.items())[0]
                bid_price_2, bid_volume_2 = list(buy_orders_copy.items())[1]

                
                ask_price_1_last = self.asks[product][-2]
                bid_price_1_last = self.bids[product][-2]
            
            
                
                acceptable_price = self.make_prediction(bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
                                                         bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
                                                         mid_price_3, mid_price_4, mid_price_5, 
                                                         bid_price_1_last, ask_price_1_last)
                           

                print("Acc. price: ", acceptable_price)
                
                
                max_size = 9999

                ########################
                ### TAKERS STARFRUIT ###     
                ########################
                
                ###BUY###
                for ask, vol in sell_orders.items():
                    if  ask < acceptable_price and rem_buy > 0:
                        size = min(-vol, rem_buy, max_size)
                        rem_buy -= size
                        new_position += size
                        assert(size >= 0)
                        product_orders.append(Order(product, ask, size))
                        print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
                    if (ask == acceptable_price and rem_buy > max_position):
                        size = min(-vol, rem_buy, rem_buy - max_position, max_size)
                        rem_buy -= size
                        new_position += size
                        assert(size >= 0)
                        product_orders.append(Order(product, ask, size))
                        print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
                    
                
                ###SELL###
                for bid, vol in buy_orders.items():
                    if  bid > acceptable_price and rem_sell > 0:
                        size = min(vol, rem_sell, max_size)
                        rem_sell -= size
                        new_position -= size
                        assert(size >= 0)
                        product_orders.append(Order(product, bid, -size))
                        print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)
                    if (bid == acceptable_price and rem_sell > max_position):
                        size = min(vol, rem_sell, rem_sell - max_position, max_size)
                        rem_sell -= size
                        new_position -= size
                        assert(size >= 0)
                        product_orders.append(Order(product, bid, -size))
                        print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)
                        
                        
                ########################
                ### MAKERS STARFRUIT ###     
                ########################
                
             ### BUY ###
             
                if new_position < 0 and rem_buy > 0:
                    size = min(rem_buy, max_size, -new_position)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))

                if new_position > 15 and rem_buy > 0:
                    size = min(rem_buy, max_size)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 2), size))

                if rem_buy > 0:
                    size = min(rem_buy, max_size)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))
                    
                ############
                
                ### SELL ###
                
                if new_position > 0 and rem_sell > 0:
                    size = min(rem_sell, max_size, new_position)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))

                if new_position < -15 and rem_sell > 0:
                    size = min(rem_sell, max_size)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 2), -size))

                if rem_sell > 0:
                    size = min(rem_sell, max_size)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))
            
            ############                


            #############################################################################
            ########################## AMETHYSTS ########################################
            #############################################################################
            
            if product == "AMETHYSTS":
            
                acceptable_price = 10000
                
                max_size = 9999
                new_position = position
                
                rem_buy = max_position - position
                rem_sell = max_position + position
                
                
                
                ########################
                ### TAKERS AMETHYSTS ###     
                ########################
                   
                ###-BUYS-###
                for ask, vol in sell_orders.items():
                    if  ask < acceptable_price and rem_buy > 0:
                        size = min(-vol, rem_buy, max_size)
                        rem_buy -= size
                        new_position += size
                        assert(size >= 0)
                        product_orders.append(Order(product, ask, size))
                    if (ask == acceptable_price and rem_buy > max_position):
                        size = min(-vol, rem_buy, rem_buy - max_position, max_size)
                        rem_buy -= size
                        new_position += size
                        assert(size >= 0)
                        product_orders.append(Order(product, ask, size))
                        
                ###-SELLS-###        
                for bid, vol in buy_orders.items():
                    if  bid > acceptable_price and rem_sell > 0:
                        size = min(vol, rem_sell, max_size)
                        rem_sell -= size
                        new_position -= size
                        assert(size >= 0)
                        product_orders.append(Order(product, bid, -size))
                    if (bid == acceptable_price and rem_sell > max_position):
                        size = min(vol, rem_sell, rem_sell - max_position, max_size)
                        rem_sell -= size
                        new_position -= size
                        assert(size >= 0)
                        product_orders.append(Order(product, bid, -size))
                        
                ########################
                ### MAKERS AMETHYSTS ###
                ########################
                
             ### BUY ###
             
                if new_position < 0 and rem_buy > 0:
                    size = min(rem_buy, max_size, -new_position)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))

                if new_position > 15 and rem_buy > 0:
                    size = min(rem_buy, max_size)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 2), size))

                if rem_buy > 0:
                    size = min(rem_buy, max_size)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))
                    
                ############
                
                ### SELL ###
                
                if new_position > 0 and rem_sell > 0:
                    size = min(rem_sell, max_size, new_position)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))

                if new_position < -15 and rem_sell > 0:
                    size = min(rem_sell, max_size)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 2), -size))

                if rem_sell > 0:
                    size = min(rem_sell, max_size)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))
            
            ############

            #############################################################################
            #############################################################################
            #############################################################################


            result[product] = product_orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data






