import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import pandas as pd
class Trader:
	def run(self, state):
		result = {}
		trader_data  = ''
		conversions = 0
		time = state.timestamp
		if time == 0:
			od = state.order_depths['ORCHIDS']
			buy_orders = list(od.buy_orders.items())
			buy_orders.sort(key = lambda x:x[0], reverse = True)
			sell_orders = list(od.sell_orders.items())
			sell_orders.sort(key = lambda x: x[0])
			best_bid = buy_orders[0][0]
			best_ask = sell_orders[0][0]
			result['ORCHIDS'] = [Order('ORCHIDS',best_bid,-2)]
			print(state.observations.conversionObservations['ORCHIDS'].bidPrice)
			print(state.observations.conversionObservations['ORCHIDS'].askPrice)
			print(state.observations.conversionObservations['ORCHIDS'].importTariff)
			print(state.observations.conversionObservations['ORCHIDS'].exportTariff)
			print(state.observations.conversionObservations['ORCHIDS'].transportFees)
            
		if time == 100:
			conversions = 2
			print(state.observations.conversionObservations['ORCHIDS'].bidPrice)
			print(state.observations.conversionObservations['ORCHIDS'].askPrice)
			print(state.observations.conversionObservations['ORCHIDS'].importTariff)
			print(state.observations.conversionObservations['ORCHIDS'].exportTariff)
			print(state.observations.conversionObservations['ORCHIDS'].transportFees)
		return result, conversions, trader_data
		# pnl = qty*( local best_bid at ts 0) - qty*(conversion best_ask at ts = 100) - qty*(import tariff) - qty*(transport fees)
        
        
        
        
        
        
        
        
            best_bid = buy_orders[0][0]
            best_ask = sell_orders[0][0]

            conversion_bid = state.observations.conversionObservations[product].bidPrice
            conversion_ask = state.observations.conversionObservations[product].askPrice
            import_tariff = state.observations.conversionObservations[product].importTariff
            export_tariff = state.observations.conversionObservations[product].exportTariff
            transport_fees = state.observations.conversionObservations[product].transportFees

            print("####### TRADING:  ", product, "###########")
            print(order_depth.sell_orders)
            print(order_depth.buy_orders, "\n")
            print("  Position:   ", position, "   ")
            print(" $$$ Conversions are: ", conversion_bid, conversion_ask, import_tariff, export_tariff, transport_feed, " $$$ ")

            if timestamp == 100:
                conversions = 2

            product_orders.append(Order('ORCHIDS', best_bid, -2))
        