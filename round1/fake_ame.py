import json
import math
import collections
from collections import defaultdict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List

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

    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}

    def __init__(self):
        self.prices = {"PRODUCT1": [9], 
                       "PRODUCT2": [150],
                       "AMETHYSTS" : [10000],
                       "STARFRUIT" : []}  # Initialize prices for each symbol
                       


        self.maximum_positions = {"PRODUCT1": 20, 
                                  "PRODUCT2": 20,
                                  "AMETHYSTS": 20,
                                  "STARFRUIT": 20}  # Maximum positions for each symbol
                                  
        self.bids = {"PRODUCT1": [], 
                          "PRODUCT2": [],
                          "AMETHYSTS": [],
                          "STARFRUIT": []}  # Bids history
                          
        self.asks = {"PRODUCT1": [], 
                          "PRODUCT2": [],
                          "AMETHYSTS": [],
                          "STARFRUIT": []}  # Asks history
                                  
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
        
    def compute_orders_AMETHYSTS(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1
        
        print("Best ask: ", best_buy_pr, "    ")
        print("Best ask amount: ", buy_vol, "    ")   
        
        print("Best ask: ", best_sell_pr, "    ")
        print("Best ask amount: ", sell_vol, "    ")   

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                print("Buy order details:", "Product: AMETHYSTS ", "Price:", ask, "Amount:", order_for,  "    ")
                

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            bidd = min(undercut_buy + 1, acc_bid-1)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            print("Buy order details:", "Product: AMETHYSTS ", "Price:", bidd, "Amount:", num,  "    ")
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            bidd = min(undercut_buy - 1, acc_bid-1)
            print("Buy order details:", "Product: AMETHYSTS ", "Price:", bidd, "Amount:", num,  "    ")
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            print("Buy order details:", "Product: AMETHYSTS ", "Price:", bid_pr, "Amount:", num,  "    ")
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                print("Sell order details:", "Product: AMETHYSTS ", "Price:", bid, "Amount:", order_for,  "    ")

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            askk = max(undercut_sell-1, acc_ask+1)
            print("Sell order details:", "Product: AMETHYSTS ", "Price:", askk, "Amount:", num,  "    ")
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            askk = max(undercut_sell+1, acc_ask+1)
            print("Sell order details:", "Product: AMETHYSTS ", "Price:", askk, "Amount:", num,  "    ")
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            print("Sell order details:", "Product: AMETHYSTS ", "Price:", sell_pr, "Amount:", num,  "    ")
            cpos += num

        return orders
        
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

        

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        
        timestamp = state.timestamp

        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            
            sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
            
            product_orders = []
            
            sell_vol, best_sell_pr = self.values_extract(sell_orders)
            buy_vol, best_buy_pr = self.values_extract(buy_orders, 1)
            
            # Define positions and maximum positions for the current symbol
            position = state.position.get(product, 0)
            max_position = self.maximum_positions.get(product, 0)
            
            rem_buy = max_position - position
            rem_sell = position + max_position
            

            print("####### TRADING:  ", product, "###########")
            print(order_depth.sell_orders)
            print(order_depth.buy_orders, "\n")
            print("  Position:   ", position, "   ")

            
            ########### UPDATE HISTORY
            

            avg_price_update = (best_sell_pr + best_buy_pr) / 2
            bid_update = best_buy_pr
            ask_update = best_sell_pr
            
            if timestamp > 0:
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
                sma = sum(self.prices[product][-20:]) / len(self.prices[product][-20:])
                acceptable_price = round(sma)
                
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
                    product_orders.append(Order(product, min(best_buy_pr + 2, acceptable_price - 1), size))

                if new_position > 15 and rem_buy > 0:
                    size = min(rem_buy, max_size)
                    rem_buy -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, min(best_buy_pr, acceptable_price - 1), size))

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
                    product_orders.append(Order(product, max(best_sell_pr - 2, acceptable_price + 1), -size))

                if new_position < -15 and rem_buy > 0:
                    size = min(rem_sell, max_size)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr, acceptable_price + 1), -size))

                if rem_buy > 0:
                    size = min(rem_sell, max_size)
                    rem_sell -= size
                    assert(size >= 0)
                    product_orders.append(Order(product, max(best_sell_pr - 1, acceptable_price + 1), -size))
            
            ############               

            def compute_orders(self, product, order_depth, acc_bid, acc_ask):

                if product == "AMETHYSTS":
                    return self.compute_orders_AMETHYSTS(product, order_depth, acc_bid, acc_ask)
                if product == "STARFRUIT":
                    return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])


            for product in ['AMETHYSTS']:
                order_depth: OrderDepth = state.order_depths[product]
                orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
                result[product] += orders
            
            ############

            #############################################################################
            #############################################################################
            #############################################################################


            result[product] = product_orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data






