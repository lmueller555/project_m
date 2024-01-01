import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date', inplace=True)
        data['Company'] = data['Company'].str.replace(
            r'\d+\.', '', regex=True).str.strip()
        return data

    def pivot_data(self):
        return self.data.pivot(index='Date', columns='Company', values='Close/Last')


# Define parameters in a dictionary
parameters = {
    "momentum_threshold": 0.025,
    "mean_rev_threshold": 0.025,
    "swing_threshold": 0.05,
    "entry_threshold": 0.05,
    "exit_threshold": 0.05,
    "max_trade_quantity": 15,
    "historical_window": 60,
    "correlation_threshold": 0.90,
    "negative_correlation_threshold": -0.65,
    "evaluation_interval": 30
    # "increment_factor": 0.01
}

#731.17%


class PairAnalyzer:
    def __init__(self, parameters):
        self.parameters = parameters

    def identify_pairs(self, correlation_matrix):
        # Flatten the correlation matrix and sort by correlation
        flattened = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                pair = (
                    correlation_matrix.columns[i], correlation_matrix.columns[j])
                correlation = correlation_matrix.iloc[i, j]
                flattened.append((pair, correlation))

        flattened.sort(key=lambda x: x[1])

        # Print the top 3 most negatively correlated
        # print("Top 3 most negatively correlated pairs:")
        # for pair, correlation in flattened[:3]:
        #     print(pair, correlation)

        # Print the top 3 most positively correlated
        # print("Top 3 most positively correlated pairs:")
        # for pair, correlation in flattened[-3:]:
        #     print(pair, correlation)

        # Identifying pairs based on the threshold
        pairs = [pair for pair, correlation in flattened if correlation >
                 self.parameters['correlation_threshold']]
        return pairs

    def calculate_normal_ratios(self, pivoted_data, pair):
        daily_ratios = pivoted_data[pair[0]] / pivoted_data[pair[1]]
        return daily_ratios[-self.parameters['historical_window']:].mean()

    @staticmethod
    def reformat_pair_names(pair):
        return tuple(re.sub(r'\d+\.', '', stock).strip() for stock in pair)


class PortfolioManager:
    def __init__(self, initial_capital, parameters):
        self.portfolio = self.initialize_portfolio(initial_capital)
        self.parameters = parameters

    def initialize_portfolio(self, initial_capital):
        return {'cash': initial_capital, 'stocks': {}}

    def buy_stock(self, stock, price):
        quantity = min(
            self.parameters['max_trade_quantity'], self.portfolio['cash'] // price)
        trade_info = {'stock': stock, 'action': 'buy', 'quantity': 0, 'price': price}

        if quantity > 0 and stock not in self.portfolio['stocks']:
            self.portfolio['stocks'][stock] = {
                'quantity': 0, 'average_buying_price': 0}

        if quantity > 0:
            total_cost = price * quantity
            total_quantity = self.portfolio['stocks'][stock]['quantity'] + quantity
            average_price = ((self.portfolio['stocks'][stock]['average_buying_price'] *
                              self.portfolio['stocks'][stock]['quantity']) + total_cost) / total_quantity

            self.portfolio['cash'] -= total_cost
            self.portfolio['stocks'][stock]['quantity'] = total_quantity
            self.portfolio['stocks'][stock]['average_buying_price'] = average_price

            trade_info['quantity'] = quantity

        return trade_info

    def sell_stock(self, stock, price):
        trade_info = {'stock': stock, 'action': 'sell', 'quantity': 0, 'price': price}

        if stock in self.portfolio['stocks']:
            quantity = min(
                self.parameters['max_trade_quantity'], self.portfolio['stocks'][stock]['quantity'])
            if quantity > 0:
                self.portfolio['cash'] += price * quantity
                self.portfolio['stocks'][stock]['quantity'] -= quantity

                if self.portfolio['stocks'][stock]['quantity'] == 0:
                    del self.portfolio['stocks'][stock]

                trade_info['quantity'] = quantity

        return trade_info

    # Additional methods for portfolio management can be added here, if needed.


class TradingStrategy:
    def __init__(self, portfolio_manager, parameters):
        self.portfolio_manager = portfolio_manager
        self.parameters = parameters
        self.trading_data = []

    def momentum_trading(self, row):
        stock = row['Company'].strip()
        price = row['Close/Last']
        prediction = row['Prediction']
        trade_info = None

        if prediction > price * (1 + self.parameters['momentum_threshold']):
            trade_info = self.portfolio_manager.buy_stock(stock, price)
        elif prediction < price * (1 - self.parameters['momentum_threshold']):
            trade_info = self.portfolio_manager.sell_stock(stock, price)

        if trade_info and trade_info['quantity'] > 0:
            trade_info['strategy'] = 'Momentum Trading'
            self.trading_data.append(trade_info)

    def mean_reversion(self, row):
        stock = row['Company'].strip()
        price = row['Close/Last']
        trade_info = None

        if row['Close/Last'] < row['30d MA'] * (1 - self.parameters['mean_rev_threshold']):
            trade_info = self.portfolio_manager.buy_stock(stock, price)
        elif row['Close/Last'] > row['30d MA'] * (1 + self.parameters['mean_rev_threshold']):
            trade_info = self.portfolio_manager.sell_stock(stock, price)

        if trade_info and trade_info['quantity'] > 0:
            trade_info['strategy'] = 'Mean Reversion'
            self.trading_data.append(trade_info)

    def swing_trading(self, row):
        stock = row['Company'].strip()
        price = row['Close/Last']
        trade_info = None

        if row['Prediction'] > price * (1 + self.parameters['swing_threshold']):
            trade_info = self.portfolio_manager.buy_stock(stock, price)
        elif row['Prediction'] < price * (1 - self.parameters['swing_threshold']):
            trade_info = self.portfolio_manager.sell_stock(stock, price)

        if trade_info and trade_info['quantity'] > 0:
            trade_info['strategy'] = 'Swing Trading'
            self.trading_data.append(trade_info)

    def pair_trading(self, aggregated_day_data, pairs, normal_ratios, current_correlation_matrix):
        for pair in pairs:
            stock1, stock2 = pair
            trade_info = None

            if stock1 in aggregated_day_data and stock2 in aggregated_day_data:
                stock1_data = aggregated_day_data[stock1]
                stock2_data = aggregated_day_data[stock2]
                current_ratio = stock1_data['Close/Last'] / stock2_data['Close/Last']
                normal_ratio = normal_ratios[pair]

                # Buy logic
                if current_ratio < normal_ratio * (1 - self.parameters['entry_threshold']):
                    if self.portfolio_manager.portfolio['cash'] >= stock1_data['Close/Last']:
                        trade_info = self.portfolio_manager.buy_stock(stock1, stock1_data['Close/Last'])
                        if trade_info['quantity'] > 0:
                            trade_info['strategy'] = 'Pair Trading'
                            self.trading_data.append(trade_info)

                elif current_ratio > normal_ratio * (1 + self.parameters['entry_threshold']):
                    if self.portfolio_manager.portfolio['cash'] >= stock2_data['Close/Last']:
                        trade_info = self.portfolio_manager.buy_stock(stock2, stock2_data['Close/Last'])
                        if trade_info['quantity'] > 0:
                            trade_info['strategy'] = 'Pair Trading'
                            self.trading_data.append(trade_info)

                # Consolidated sell and buy logic
                for stock, stock_data in [(stock1, stock1_data), (stock2, stock2_data)]:
                    sell_condition = False

                    # Check if the stock meets the mean reversion sell condition
                    if normal_ratio * (1 - self.parameters['exit_threshold']) < current_ratio < normal_ratio * (1 + self.parameters['exit_threshold']):
                        sell_condition = True

                    # Check for 5% drop below 30-day MA
                    elif stock_data['Close/Last'] < stock_data['30d MA'] * 0.95:
                        sell_condition = True

                    if sell_condition:
                        if stock in self.portfolio_manager.portfolio and self.portfolio_manager.portfolio[stock] > 0:
                            trade_info = self.portfolio_manager.sell_stock(stock, stock_data['Close/Last'])
                            if trade_info['quantity'] > 0:
                                trade_info['strategy'] = 'Pair Trading'
                                self.trading_data.append(trade_info)

                    # Identifying the most negatively correlated stock
                    most_negatively_correlated = current_correlation_matrix[stock].idxmin()
                    correlation_value = current_correlation_matrix[stock][most_negatively_correlated]
                    
                    if correlation_value <= self.parameters['negative_correlation_threshold']:
                        if most_negatively_correlated not in pair:
                            negatively_correlated_stock_data = aggregated_day_data[most_negatively_correlated]

                            if self.portfolio_manager.portfolio['cash'] >= negatively_correlated_stock_data['Close/Last']:
                                trade_info = self.portfolio_manager.buy_stock(most_negatively_correlated, negatively_correlated_stock_data['Close/Last'])
                                if trade_info['quantity'] > 0:
                                    trade_info['strategy'] = 'Pair Trading'
                                    self.trading_data.append(trade_info)


class PerformanceAnalytics:
    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def calculate_portfolio_value(portfolio, current_prices):
        total_stock_value = 0
        for stock, info in portfolio['stocks'].items():
            if stock in current_prices:
                total_stock_value += current_prices[stock] * info['quantity']
            else:
                print(
                    f"Debug: {stock} not found in current_prices. Available keys: {list(current_prices.keys())}")
        return portfolio['cash'] + total_stock_value

    @staticmethod
    def calculate_roi(initial_capital, current_value):
        return (current_value - initial_capital) / initial_capital

    @staticmethod
    def calculate_maximum_drawdown(portfolio_values):
        peak = max(portfolio_values)
        trough = min(portfolio_values)
        return (peak - trough) / peak

    def adjust_parameters_based_on_roi(self, roi, last_roi):
        increment_factor = self.parameters['increment_factor']

        if roi > last_roi:  # If ROI has increased
            self.parameters['max_trade_quantity'] = int(
                self.parameters['max_trade_quantity'] * (1 + increment_factor))
        elif roi < last_roi:  # If ROI has decreased
            self.parameters['max_trade_quantity'] = max(
                1, int(self.parameters['max_trade_quantity'] * (1 - increment_factor)))
        # Additional logic for adjusting other strategy thresholds can be added here


# Initialize classes with the dataset file path
data_processor = DataProcessor(
    'C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv')
pair_analyzer = PairAnalyzer(parameters)

# Initialize PortfolioManager once outside the generation loop
portfolio_manager = PortfolioManager(50000, parameters)
trading_strategy = TradingStrategy(portfolio_manager, parameters)
performance_analytics = PerformanceAnalytics(parameters)

daily_portfolio_values_last_gen = []
evaluation_interval = parameters['evaluation_interval']
num_generations = 1
generation_rois = []

unique_trading_days = pd.to_datetime(data_processor.data['Date'].unique())
unique_trading_days = np.sort(unique_trading_days)

for generation in range(num_generations):
    print(f"\nStarting Generation {generation + 1}")
    portfolio_values = []

    for trading_day in unique_trading_days:
        trading_day = pd.Timestamp(trading_day)
        print(f"Processing trading day: {trading_day.strftime('%Y-%m-%d')}")

        historical_data = data_processor.data[data_processor.data['Date'] <= trading_day]
        pivoted_historical_data = historical_data.pivot(
            index='Date', columns='Company', values='Close/Last')

        current_correlation_matrix = pivoted_historical_data.corr()
        current_pairs = pair_analyzer.identify_pairs(
            current_correlation_matrix)
        formatted_current_pairs = [
            PairAnalyzer.reformat_pair_names(pair) for pair in current_pairs]
        current_normal_ratios = {pair: pair_analyzer.calculate_normal_ratios(
            pivoted_historical_data, pair) for pair in formatted_current_pairs}

        daily_data = data_processor.data[data_processor.data['Date']
                                         == trading_day]
        if daily_data.empty:
            continue

        # Update current_prices for the day
        current_prices = {row['Company'].strip(): row['Close/Last']
                          for _, row in daily_data.iterrows()}

        aggregated_day_data = {row['Company'].strip(
        ): row for _, row in daily_data.iterrows()}
        for _, row in aggregated_day_data.items():
            trading_strategy.momentum_trading(row)
            trading_strategy.mean_reversion(row)
            trading_strategy.swing_trading(row)

        trading_strategy.pair_trading(
            aggregated_day_data, formatted_current_pairs, current_normal_ratios, current_correlation_matrix)

        current_value = performance_analytics.calculate_portfolio_value(
            portfolio_manager.portfolio, current_prices)
        portfolio_values.append(current_value)

        if len(portfolio_values) % evaluation_interval == 0:
            current_roi = performance_analytics.calculate_roi(
                50000, current_value)
            max_drawdown = performance_analytics.calculate_maximum_drawdown(
                portfolio_values[-evaluation_interval:])
            print(
                f"Day {trading_day.strftime('%Y-%m-%d')}: ROI: {current_roi:.2%}, Max Drawdown: {max_drawdown:.2%}")

    # Just before the final valuation
    # print(f"Final Portfolio before valuation: {portfolio_manager.portfolio}")
    # print(f"Final Current Prices: {current_prices}")

    final_value = performance_analytics.calculate_portfolio_value(
        portfolio_manager.portfolio, current_prices)
    final_roi = performance_analytics.calculate_roi(50000, final_value)

    print(f"End of Generation {generation + 1}: Final ROI: {final_roi:.2%}")
    print(
        f"Total Portfolio Value at End of Generation {generation + 1}: ${final_value:,.2f}")
    print(
        f"Total Profit at End of Generation {generation + 1}: ${final_value - 50000:,.2f}")

    if generation == num_generations - 1:
        daily_portfolio_values_last_gen = portfolio_values.copy()

# After the loop over generations
final_portfolio = portfolio_manager.portfolio

# Extracting stock names and their quantities from the final portfolio
stock_names = list(final_portfolio['stocks'].keys())
quantities = [final_portfolio['stocks'][stock]['quantity']
              for stock in stock_names]


# Counting trades and calculating profits per strategy
trade_counts = {'Momentum Trading': 0, 'Mean Reversion': 0, 'Swing Trading': 0, 'Pair Trading': 0}
strategy_profits = {'Momentum Trading': 0, 'Mean Reversion': 0, 'Swing Trading': 0, 'Pair Trading': 0}

for trade in trading_strategy.trading_data:
    strategy = trade['strategy']
    stock = trade['stock']
    action = trade['action']
    quantity = trade['quantity']
    price = trade['price']

    # Update trade counts
    trade_counts[strategy] += 1

    # Calculate profit/loss
    if stock in final_portfolio['stocks']:
        stock_data = final_portfolio['stocks'][stock]
        if action == 'sell':
            profit = (price - stock_data['average_buying_price']) * quantity
            strategy_profits[strategy] += profit
        # For buys, we will consider profit calculation at the end of the generation

# Adding unrealized profits for the stocks still held
for stock, stock_data in final_portfolio['stocks'].items():
    current_price = current_prices[stock]
    unrealized_profit = (current_price - stock_data['average_buying_price']) * stock_data['quantity']
    # Assuming equal distribution of unrealized profit across strategies
    for strategy in strategy_profits:
        strategy_profits[strategy] += unrealized_profit / len(strategy_profits)

# Plotting daily portfolio values for the last generation
plt.figure(figsize=(14, 7))
plt.plot(daily_portfolio_values_last_gen)
plt.title('Daily Portfolio Value')
plt.xlabel('Day')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()

# Bar graph for the number of trades per strategy
plt.figure(figsize=(14, 7))
plt.bar(trade_counts.keys(), trade_counts.values(), color='skyblue')
plt.title('Number of Trades by Strategy')
plt.xlabel('Strategy')
plt.ylabel('Number of Trades')
plt.show()

# Bar graph for the profit per strategy
plt.figure(figsize=(14, 7))
plt.bar(strategy_profits.keys(), strategy_profits.values(), color='lightgreen')
plt.title('Profit by Strategy')
plt.xlabel('Strategy')
plt.ylabel('Profit')
plt.show()

# Creating the bar graph
plt.figure(figsize=(14, 7))
plt.bar(stock_names, quantities, color='blue')
plt.xlabel('Stocks')
plt.ylabel('Quantity')
plt.title('Portfolio Composition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv