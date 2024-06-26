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

    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    
    begin_diff_dip = -9999999
    begin_diff_bag = -9999999
    begin_bag_price = -9999999
    begin_dip_price = -9999999

    std = 25
    basket_std = 70

    
    def __init__(self):
        self.prices = {"PRODUCT1": [9], 
                       "PRODUCT2": [150],
                       "AMETHYSTS" : [10000],
                       "STARFRUIT" : [],
                       "ORCHIDS" : [],
                       "TOTAL_VOL_ORCHIDS" : [],
                       "SPREAD" : []}  # Initialize prices for each symbol


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

        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods: # Get prices and volumes
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
                
        spread = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES']
        self.prices['SPREAD'].append(spread)

        basket_std = 76.4
        ma_spread = 379.5
        std_coeff = 0.5
        window = 200

        # if len(self.prices['SPREAD']) > 5:
        #     ma_spread = pd.Series(self.prices['SPREAD']).rolling(5).mean().iloc[-1]

        # if len(self.prices['SPREAD']) > window:
        #     ma_spread = pd.Series(self.prices['SPREAD']).rolling(window).mean().iloc[-1]
        #     basket_std = pd.Series(self.prices['SPREAD']).rolling(window).std().iloc[-1]
                   
        print("NOW TRADING GIFT BASKETS ")

        res_buy = spread - ma_spread
        res_sell = spread - ma_spread

        trade_at = basket_std*std_coeff

        rem_buy = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET'] # Pozitivan broj
        rem_sell = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET'] # Pozitivan broj


        print(" $$ NOW TRADING GIFT BASKETS ")
        print(" $$ CURRENT POSITION: ", self.position['GIFT_BASKET'])
        print(" $$ SPREAD FROM MEAN (USING MID PRICES): ", res_buy)

        ########### get desirable positions for components

        ##################################################

        if res_sell > trade_at:
            print("SPREAD IS TOO BIG!! WE WILL SELL GIFT BASKETS")
            vol = rem_sell
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                print("Sell order details:", "Product: GIFT_BASKET", "Price:", worst_buy['GIFT_BASKET'], "Amount:", -vol)
        elif res_buy < -trade_at:
            print("SPREAD IS TOO LOW!! WE WILL BUY GIFT BASKETS")
            vol = rem_buy
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                print("Sell order details:", "Product: GIFT_BASKED", "Price:", worst_sell['GIFT_BASKET'], "Amount:", vol)



        return orders

        

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        
        timestamp = state.timestamp

        result = {}
        
        ords = self.compute_orders_basket(state.order_depths, state.position)
        result['GIFT_BASKET'] = ords['GIFT_BASKET']
        result['STRAWBERRIES'] = ords['STRAWBERRIES']
        result['CHOCOLATE'] = ords['CHOCOLATE']
        result['ROSES'] = ords['ROSES']
        
        # for product in ["ORCHIDS"]:
        #     product_orders = []
        #     order_depth = state.order_depths[product]
            
        #     sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items(), reverse=True))
        #     buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items()))
            
        #     position = state.position.get(product, 0)
        #     max_position = self.maximum_positions.get(product, 0)
        #     new_position = position
            
        #     rem_buy = max_position - position
        #     rem_sell = max_position + position
            
        #     sell_vol, best_ask = self.values_extract(sell_orders)
        #     buy_vol, best_bid = self.values_extract(buy_orders, 1)

        #     conversion_bid = state.observations.conversionObservations[product].bidPrice
        #     conversion_ask = state.observations.conversionObservations[product].askPrice
        #     import_tariff = state.observations.conversionObservations[product].importTariff
        #     export_tariff = state.observations.conversionObservations[product].exportTariff
        #     transport_fees = state.observations.conversionObservations[product].transportFees
            
        #     ########### UPDATE HISTORY ##############################
            

        #     avg_price_update = (best_ask + best_bid) / 2
        #     bid_update = best_bid
        #     ask_update = best_ask
            
        #     if len(self.prices[product]) > 0:
        #         # Append values to the lists
        #         self.prices[product].append(avg_price_update)
        #         self.bids[product].append(bid_update)
        #         self.asks[product].append(ask_update)

        #         # Remove the first element of each list
        #         self.prices[product] = self.prices[product][1:]
        #         self.bids[product] = self.bids[product][1:]
        #         self.asks[product] = self.asks[product][1:]
        #     else:
        #         self.prices[product].extend([avg_price_update, avg_price_update, avg_price_update, avg_price_update, avg_price_update])
        #         self.bids[product].extend([bid_update, bid_update])
        #         self.asks[product].extend([ask_update, bid_update])
            
        #     ###########################################################
        #     ###########################################################       
        #     ###########################################################
            
        #     # Calculate SMA using only the last 5 prices
        #     sma = sum(self.prices[product][-5:]) / 5
        #     acceptable_price = round(sma, 1)

        #     print("####### TRADING:  ", product, "###########")
        #     print(sell_orders)
        #     print(buy_orders, "\n")
        #     print("Highest bid: ", best_bid)
        #     print("Lowest ask: ", best_ask)
        #     print("Acceptable price: ", acceptable_price)
        #     print("  Position:   ", position, "   ")
        #     print(" $$$ Conversions are: ", conversion_bid, conversion_ask, import_tariff, export_tariff, transport_fees, " $$$ ")
            
        #     buy_prize = -import_tariff - transport_fees # NEGATIVE, DOBIJAŠ ZA KUPOVINU
        #     sell_cost = export_tariff + transport_fees # POSITIVE , PLAĆAŠ ZA PRODAJU
            
        #     holding_fees = 0.1

            
        #     ########################
        #     ### TAKERS ORCHIDS ###     
        #     ########################
            
        #     max_size = 9999
        #     max_conversion_short = 10
        #     max_conversion = abs(position)
            
        #     ### Add costs for holding etc.... stimulate selling
            
        #     # if position > -max_conversion_short:
        #         # product_orders.append(Order(product, best_bid, -position - max_conversion_short))
                
        #     ####### MEGA ARB
            
            
        #     true_south_ask = conversion_ask + import_tariff + transport_fees
        #     true_south_bid = conversion_bid - export_tariff - transport_fees
        #     best_local_ask = best_ask
        #     best_local_bid = best_bid
        #     #if curr_humidity < prev_humidity and curr_humidity > 80:
                        
        #     self.TOTAL_VOL_ORCHIDS += abs(position - self.LAST_POSITION)
        #     self.LAST_POSITION = position
            
        #     if(true_south_ask < best_local_ask - 1):
        #         print(" ARB SPOTTED SELL LOCAL BUY SOUTH ")
        #     if(true_south_bid > best_local_bid + 1):
        #         print(" ARB SPOTTED SELL SOUTH BUY LOCAL ")

        #     if position != 0:
        #         conversions = -position
        #         print("Conversion details:", "Product:", product, "True south ask:", true_south_ask, "Amount:", position)
                
            
        #     if(true_south_ask < best_local_ask):
        #         product_orders.append(Order(product, math.floor(true_south_ask) + 2 , -rem_sell))
        #         print("Sell order details:", "Product:", product, "Price:", math.floor(true_south_ask) + 2, "Amount:", -rem_sell)
                
                
        #     print(" &&&&&&& TOTAL ORCHIDS VOL SO FAR:" , self.TOTAL_VOL_ORCHIDS, "&&&&&&&&&")

                
           
            
        #     ################
            
        #     ###BUY###
        #     # for ask, vol in sell_orders.items():
        #         # if (ask + sell_cost < conversion_bid):
        #             # print(" BUY LOCAL SELL SOUTH ARB SPOTTED  ")
        #             # if position > 0:
        #                 # conv = min((-vol), max_conversion)
        #                 # conversions += conv
        #                 # max_conversion -= conv
        #                 # vol += conv
        #                 # print("Conversion details:", "Product:", product, "Conv_bid:", conversion_bid, " Costs:  ",  sell_costs, "Local ask: ", ask, "Amount:", conv)
        #                 # size = conv
        #                 # rem_buy -= size
        #                 # new_position += size
        #                 # assert(size >= 0)
        #                 # product_orders.append(Order(product, ask, size))
        #                 # print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
        #         # if  ask < acceptable_price and rem_buy > 0 and vol < 0:
        #             # size = min(-vol, rem_buy, max_size)
        #             # rem_buy -= size
        #             # new_position += size
        #             # assert(size >= 0)
        #             # product_orders.append(Order(product, ask, size))
        #             # print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
        #         # if (ask == acceptable_price and rem_buy > max_position and vol < 0):
        #             # size = min(-vol, rem_buy, rem_buy - max_position, max_size)
        #             # rem_buy -= size
        #             # new_position += size
        #             # assert(size >= 0)
        #             # product_orders.append(Order(product, ask, size))
        #             # print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
                
            
        #     ###SELL###
        #     # for bid, vol in buy_orders.items():
        #         # if (bid + buy_prize > conversion_ask):
        #             # print("  SELL LOCAL BUY SOUTH ARB SPOTTED  ")
        #             # if position < 0:
        #                 # conv = min(vol, max_conversion)
        #                 # conversions += conv
        #                 # max_conversion -= conv
        #                 # vol -= conv
        #                 # print("Conversion details:", "Product:", product, "Conv_ask:", conversion_ask, " Prize:  ",  buy_prize, "Local bid: ", bid, "Amount:", conv)
        #                 # size = conv
        #                 # rem_sell -= size
        #                 # new_position -= size
        #                 # assert(size >= 0)
        #                 # product_orders.append(Order(product, bid, -size))
        #                 # print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)                
        #         # if  bid > acceptable_price and rem_sell > 0 and vol > 0:
        #             # size = min(vol, rem_sell, max_size)
        #             # rem_sell -= size
        #             # new_position -= size
        #             # assert(size >= 0)
        #             # product_orders.append(Order(product, bid, -size))
        #             # print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)
        #         # if (bid == acceptable_price and rem_sell > max_position and vol > 0):
        #             # size = min(vol, rem_sell, rem_sell - max_position, max_size)
        #             # rem_sell -= size
        #             # new_position -= size
        #             # assert(size >= 0)
        #             # product_orders.append(Order(product, bid, -size))
        #             # print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)            
            

            


        #     result[product] = product_orders
            
            
            
            
        ######################################################################################


        # for product in ["AMETHYSTS", "STARFRUIT"]:
        #     order_depth = state.order_depths[product]
            
        #     sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        #     buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
            
        #     product_orders = []
            
        #     sell_vol, best_sell_pr = self.values_extract(sell_orders)
        #     buy_vol, best_buy_pr = self.values_extract(buy_orders, 1)
            
        #     # Define positions and maximum positions for the current symbol
        #     position = state.position.get(product, 0)
        #     max_position = self.maximum_positions.get(product, 0)
        #     new_position = position
            
        #     rem_buy = max_position - position
        #     rem_sell = position + max_position
            

        #     print("####### TRADING:  ", product, "###########")
        #     print(order_depth.sell_orders)
        #     print(order_depth.buy_orders, "\n")
        #     print("Highest bid: ", best_buy_pr)
        #     print("Lowest ask: ", best_sell_pr)
        #     print("  Position:   ", position, "   ")

            
        #     ########### UPDATE HISTORY
            

        #     avg_price_update = (best_sell_pr + best_buy_pr) / 2
        #     bid_update = best_buy_pr
        #     ask_update = best_sell_pr
            
        #     if len(self.prices[product]) > 0:
        #         # Append values to the lists
        #         self.prices[product].append(avg_price_update)
        #         self.bids[product].append(bid_update)
        #         self.asks[product].append(ask_update)

        #         # Remove the first element of each list
        #         self.prices[product] = self.prices[product][1:]
        #         self.bids[product] = self.bids[product][1:]
        #         self.asks[product] = self.asks[product][1:]
        #     else:
        #         self.prices[product].extend([avg_price_update, avg_price_update, avg_price_update, avg_price_update, avg_price_update])
        #         self.bids[product].extend([bid_update, bid_update])
        #         self.asks[product].extend([ask_update, bid_update])
            
        #     ###########################################################
        #     ####### STARFRUIT #########################################       
        #     ###########################################################
           
            
        #     if product == "STARFRUIT":
            
        #         new_position = position
                
        #         mid_price_1 = self.prices[product][-1]
        #         mid_price_2 = self.prices[product][-2]
        #         mid_price_3 = self.prices[product][-3]
        #         mid_price_4 = self.prices[product][-4]
        #         mid_price_5 = self.prices[product][-5]
                
        #         # Make copies of the dictionaries manually
        #         sell_orders_copy = {key: value for key, value in sell_orders.items()}
        #         buy_orders_copy = {key: value for key, value in buy_orders.items()}

        #         # Ensure there are at least two items in sell_orders_copy
        #         if len(sell_orders_copy) < 2:
        #             first_sell_key, first_sell_value = next(iter(sell_orders_copy.items()))
        #             sell_orders_copy[first_sell_key + 1] = 0  # Add a second item with volume 0

        #         # Ensure there are at least two items in buy_orders_copy
        #         if len(buy_orders_copy) < 2:
        #             first_buy_key, first_buy_value = next(iter(buy_orders_copy.items()))
        #             buy_orders_copy[first_buy_key - 1] = 0  # Add a second item with volume 0

        #         # Now you can safely access the first and second items
        #         ask_price_1, ask_volume_1 = list(sell_orders_copy.items())[0]
        #         ask_price_2, ask_volume_2 = list(sell_orders_copy.items())[1]

        #         bid_price_1, bid_volume_1 = list(buy_orders_copy.items())[0]
        #         bid_price_2, bid_volume_2 = list(buy_orders_copy.items())[1]

                
        #         ask_price_1_last = self.asks[product][-2]
        #         bid_price_1_last = self.bids[product][-2]
            
            
                
        #         acceptable_price = self.make_prediction(bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
        #                                                  bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
        #                                                  mid_price_3, mid_price_4, mid_price_5, 
        #                                                  bid_price_1_last, ask_price_1_last)
                           

        #         print("Acc. price: ", acceptable_price)
                
                
        #         max_size = 9999

        #         ########################
        #         ### TAKERS STARFRUIT ###     
        #         ########################
                
        #         ###BUY###
        #         for ask, vol in sell_orders.items():
        #             if  ask < acceptable_price and rem_buy > 0:
        #                 size = min(-vol, rem_buy, max_size)
        #                 rem_buy -= size
        #                 new_position += size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, ask, size))
        #                 print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
        #             if (ask == acceptable_price and rem_buy > max_position):
        #                 size = min(-vol, rem_buy, rem_buy - max_position, max_size)
        #                 rem_buy -= size
        #                 new_position += size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, ask, size))
        #                 print("Sell order details:", "Product:", product, "Price:", ask, "Amount:", size)
                    
                
        #         ###SELL###
        #         for bid, vol in buy_orders.items():
        #             if  bid > acceptable_price and rem_sell > 0:
        #                 size = min(vol, rem_sell, max_size)
        #                 rem_sell -= size
        #                 new_position -= size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, bid, -size))
        #                 print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)
        #             if (bid == acceptable_price and rem_sell > max_position):
        #                 size = min(vol, rem_sell, rem_sell - max_position, max_size)
        #                 rem_sell -= size
        #                 new_position -= size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, bid, -size))
        #                 print("Sell order details:", "Product:", product, "Price:", bid, "Amount:", size)
                        
                        
        #         ########################
        #         ### MAKERS STARFRUIT ###     
        #         ########################
                
        #      ### BUY ###
             
        #         if new_position < 0 and rem_buy > 0:
        #             size = min(rem_buy, max_size, -new_position)
        #             rem_buy -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))

        #         if new_position > 15 and rem_buy > 0:
        #             size = min(rem_buy, max_size)
        #             rem_buy -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 2), size))

        #         if rem_buy > 0:
        #             size = min(rem_buy, max_size)
        #             rem_buy -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))
                    
        #         ############
                
        #         ### SELL ###
                
        #         if new_position > 0 and rem_sell > 0:
        #             size = min(rem_sell, max_size, new_position)
        #             rem_sell -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))

        #         if new_position < -15 and rem_sell > 0:
        #             size = min(rem_sell, max_size)
        #             rem_sell -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 2), -size))

        #         if rem_sell > 0:
        #             size = min(rem_sell, max_size)
        #             rem_sell -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))
            
        #     ############                


        #     #############################################################################
        #     ########################## AMETHYSTS ########################################
        #     #############################################################################
            
        #     if product == "AMETHYSTS":
            
        #         acceptable_price = 10000
                
        #         max_size = 9999
        #         new_position = position
                
        #         rem_buy = max_position - position
        #         rem_sell = max_position + position
                
                
                
        #         ########################
        #         ### TAKERS AMETHYSTS ###     
        #         ########################
                   
        #         ###-BUYS-###
        #         for ask, vol in sell_orders.items():
        #             if  ask < acceptable_price and rem_buy > 0:
        #                 size = min(-vol, rem_buy, max_size)
        #                 rem_buy -= size
        #                 new_position += size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, ask, size))
        #             if (ask == acceptable_price and rem_buy > max_position):
        #                 size = min(-vol, rem_buy, rem_buy - max_position, max_size)
        #                 rem_buy -= size
        #                 new_position += size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, ask, size))
                        
        #         ###-SELLS-###        
        #         for bid, vol in buy_orders.items():
        #             if  bid > acceptable_price and rem_sell > 0:
        #                 size = min(vol, rem_sell, max_size)
        #                 rem_sell -= size
        #                 new_position -= size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, bid, -size))
        #             if (bid == acceptable_price and rem_sell > max_position):
        #                 size = min(vol, rem_sell, rem_sell - max_position, max_size)
        #                 rem_sell -= size
        #                 new_position -= size
        #                 assert(size >= 0)
        #                 product_orders.append(Order(product, bid, -size))
                        
        #         ########################
        #         ### MAKERS AMETHYSTS ###
        #         ########################
                
        #      ### BUY ###
             
        #         if new_position < 0 and rem_buy > 0:
        #             size = min(rem_buy, max_size, -new_position)
        #             rem_buy -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))

        #         if new_position > 15 and rem_buy > 0:
        #             size = min(rem_buy, max_size)
        #             rem_buy -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 2), size))

        #         if rem_buy > 0:
        #             size = min(rem_buy, max_size)
        #             rem_buy -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, min(best_buy_pr + 1, acceptable_price - 1), size))
                    
        #         ############
                
        #         ### SELL ###
                
        #         if new_position > 0 and rem_sell > 0:
        #             size = min(rem_sell, max_size, new_position)
        #             rem_sell -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))

        #         if new_position < -15 and rem_sell > 0:
        #             size = min(rem_sell, max_size)
        #             rem_sell -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 2), -size))

        #         if rem_sell > 0:
        #             size = min(rem_sell, max_size)
        #             rem_sell -= size
        #             assert(size >= 0)
        #             product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))
            
        #     ############

        #     #############################################################################
        #     #############################################################################
        #     #############################################################################


        #     result[product] = product_orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data






