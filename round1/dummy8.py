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
    def __init__(self):
        self.prices = {"PRODUCT1": [9], 
                       "PRODUCT2": [150],
                       "AMETHYSTS" : [10000],
                       "STARFRUIT" : []}  # Initialize prices for each symbol


        self.maximum_positions = {"PRODUCT1": 20, 
                                  "PRODUCT2": 20,
                                  "AMETHYSTS": 20,
                                  "STARFRUIT": 20}  # Maximum positions for each symbol
                                  
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
        

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            product_orders = []

            print("####### TRADING:  ", product, "###########")
            print(order_depth.sell_orders)
            print(order_depth.buy_orders, "\n")
            

            # Define positions and maximum positions for the current symbol
            positions = state.position.get(product, 0)
            max_position = self.maximum_positions.get(product, 0)

            print("  Position:   ", positions, "   ")

            # Update prices for the current symbol
            if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                min_ask = min(order_depth.sell_orders.keys())
                max_bid = max(order_depth.buy_orders.keys())
                if min_ask is not None and max_bid is not None:
                    avg_price_update = (min_ask + max_bid) / 2
                else:
                    avg_price_update = self.prices[product][-1]
            
            if len(self.prices[product]) > 0:
            # Calculate Simple Moving Average (SMA) for the current trading session
                sma = sum(self.prices[product][-20:]) / len(self.prices[product][-20:])
            else:
                sma = avg_price_update
            
            acceptable_price = sma
            
            if product == "AMETHYSTS":
                acceptable_price = 10000
                
            if product == "STARFRUIT" or product == "AMETHYSTS":
                if positions < -15:
                    buy_margin = 0
                    sell_margin = 2
                elif positions < -5:
                    buy_margin = 1
                    sell_margin = 1
                elif positions < 5:
                    buy_margin = 1
                    sell_margin = 11
                elif positions < 15:
                    buy_margin = 1
                    sell_margin = 1
                else:
                    buy_margin = 2
                    sell_margin = 0
   

            print("Acc. price: ", acceptable_price)
            
            remaining_buy_position = max_position - positions
            remaining_sell_position = positions + max_position
            
            if product == "STARFRUIT":
                order_limt = 10

            # Taker sells for STARFRUIT
            if product == "STARFRUIT":
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    print("Best ask: ", best_ask, "    ")
                    print("Best ask amount: ", best_ask_amount, "    ")
                    if int(best_ask) <= acceptable_price - buy_margin:
                        clipped_best_ask_amount = min(-best_ask_amount, remaining_buy_position, order_limt)
                        product_orders.append(Order(product, best_ask, clipped_best_ask_amount))
                        remaining_buy_position -= clipped_best_ask_amount
                        print("Buy order details:", "Product:", product, "Price:", best_ask, "Amount:", clipped_best_ask_amount,  "    ")
            
            # Taker buys for STARFRUIT
            if product == "STARFRUIT":
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    print("Best bid: ", best_bid)
                    print("Best bid amount: ", best_bid_amount)
                    if int(best_bid) >= acceptable_price + sell_margin:
                        clipped_best_bid_amount = max(-best_bid_amount, -remaining_sell_position, -order_limt)
                        product_orders.append(Order(product, best_bid, clipped_best_bid_amount))
                        remaining_sell_position += clipped_best_bid_amount
                        print("Sell order details:", "Product:", product, "Price:", best_bid, "Amount:", clipped_best_bid_amount)


            #############################################################################
            

            ### TAKERS AMETHYSTS ###
            
            if product == "AMETHYSTS":
            
                acceptable_ask = acceptable_price
                acceptable_bid = acceptable_price
            
                sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
                buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

                sell_vol, best_sell_pr = self.values_extract(sell_orders)
                buy_vol, best_buy_pr = self.values_extract(buy_orders, 1)
               
                
                for ask, vol in sell_orders.items():
                    if (ask <= acceptable_bid - buy_margin) and remaining_buy_position > 0:
                        size = min(-vol, remaining_buy_position)
                        remaining_buy_position -= size
                        assert(size >= 0)
                        product_orders.append(Order(product, ask, size))
                        
                for bid, vol in buy_orders.items():
                    if (bid >= acceptable_ask + sell_margin) and remaining_sell_position > 0:
                        size = max(-vol, -remaining_sell_position)
                        remaining_sell_position += size
                        assert(size <= 0)
                        product_orders.append(Order(product, bid, size))
         
            ### MAKERS AMETHYSTS ###
            
            if product == "AMETHYSTS":
                val1buy = math.floor(remaining_buy_position/3)
                val2buy = math.floor(remaining_buy_position/3)
                
                val1sell = math.floor(remaining_sell_position/3)
                val2sell = math.floor(remaining_sell_position/3)
                
                # Place maker buy order for "AMETHYSTS"
                if positions < max_position:
                    product_orders.append(Order(product, 9997, min(val1buy, remaining_buy_position)))
                    product_orders.append(Order(product, 9998, min(val2buy, remaining_buy_position - val1buy)))
                    product_orders.append(Order(product, 9996, max(remaining_buy_position - val1buy - val2buy, 0)))

                # Place maker sell orders for AMETHYSTIS
                if positions > -max_position:
                    product_orders.append(Order(product, 10003, -min(val1sell, remaining_sell_position)))
                    product_orders.append(Order(product, 10002, -min(val2sell, remaining_sell_position - val1sell)))
                    product_orders.append(Order(product, 10004, -max(remaining_sell_position - val1sell - val2sell, 0)))                


            result[product] = product_orders
            self.prices[product].append(avg_price_update)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data






