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
        data['Company'] = data['Company'].str.replace(r'\d+\.', '', regex=True).str.strip()
        return data

    def pivot_data(self):
        return self.data.pivot(index='Date', columns='Company', values='Close/Last')


# Define parameters in a dictionary
parameters = {
    "momentum_threshold": 0.025,
    "mean_rev_threshold": 0.025,
    "swing_threshold": 0.05,
    "entry_threshold": 0.05,
    "exit_threshold": 0.03,
    "max_trade_quantity": 10,
    "historical_window": 60,
    "correlation_threshold": 0.95,
    "evaluation_interval": 30,
    "increment_factor": 0.01
}

class PairAnalyzer:
    def __init__(self, parameters):
        self.parameters = parameters

    def identify_pairs(self, correlation_matrix):
        pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.parameters['correlation_threshold']:
                    pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
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
        quantity = min(self.parameters['max_trade_quantity'], self.portfolio['cash'] // price)
        if quantity > 0 and stock not in self.portfolio['stocks']:
            self.portfolio['stocks'][stock] = {'quantity': 0, 'average_buying_price': 0}

        if quantity > 0:
            total_cost = price * quantity
            total_quantity = self.portfolio['stocks'][stock]['quantity'] + quantity
            average_price = ((self.portfolio['stocks'][stock]['average_buying_price'] *
                              self.portfolio['stocks'][stock]['quantity']) + total_cost) / total_quantity

            self.portfolio['cash'] -= total_cost
            self.portfolio['stocks'][stock]['quantity'] = total_quantity
            self.portfolio['stocks'][stock]['average_buying_price'] = average_price

            #print(f"After buying {stock}, portfolio: {self.portfolio}")


    def sell_stock(self, stock, price):
        if stock in self.portfolio['stocks']:
            quantity = min(self.parameters['max_trade_quantity'], self.portfolio['stocks'][stock]['quantity'])
            if quantity > 0:
                self.portfolio['cash'] += price * quantity
                self.portfolio['stocks'][stock]['quantity'] -= quantity

                if self.portfolio['stocks'][stock]['quantity'] == 0:
                    del self.portfolio['stocks'][stock]

                #print(f"After selling {stock}, portfolio: {self.portfolio}")

    # Additional methods for portfolio management can be added here, if needed.


class TradingStrategy:
    def __init__(self, portfolio_manager, parameters):
        self.portfolio_manager = portfolio_manager
        self.parameters = parameters

    def momentum_trading(self, row):
        stock = row['Company'].strip()
        price = row['Close/Last']
        prediction = row['Prediction']

        if prediction > price * (1 + self.parameters['momentum_threshold']):
            self.portfolio_manager.buy_stock(stock, price)
        elif prediction < price * (1 - self.parameters['momentum_threshold']):
            self.portfolio_manager.sell_stock(stock, price)

    def mean_reversion(self, row):
        stock = row['Company'].strip()
        price = row['Close/Last']
        if row['Close/Last'] < row['30d MA'] * (1 - self.parameters['mean_rev_threshold']):
            self.portfolio_manager.buy_stock(stock, price)
        elif row['Close/Last'] > row['30d MA'] * (1 + self.parameters['mean_rev_threshold']):
            self.portfolio_manager.sell_stock(stock, price)

    def swing_trading(self, row):
        stock = row['Company'].strip()
        price = row['Close/Last']
        if row['Prediction'] > price * (1 + self.parameters['swing_threshold']):
            self.portfolio_manager.buy_stock(stock, price)
        elif row['Prediction'] < price * (1 - self.parameters['swing_threshold']):
            self.portfolio_manager.sell_stock(stock, price)

    def pair_trading(self, aggregated_day_data, pairs, normal_ratios):
        for pair in pairs:
            stock1, stock2 = pair

            if stock1 in aggregated_day_data and stock2 in aggregated_day_data:
                stock1_data = aggregated_day_data[stock1]
                stock2_data = aggregated_day_data[stock2]
                current_ratio = stock1_data['Close/Last'] / stock2_data['Close/Last']
                normal_ratio = normal_ratios[pair]

                if current_ratio < normal_ratio * (1 - self.parameters['entry_threshold']):
                    if self.portfolio_manager.portfolio['cash'] >= stock1_data['Close/Last']:
                        self.portfolio_manager.buy_stock(stock1, stock1_data['Close/Last'])

                elif current_ratio > normal_ratio * (1 + self.parameters['entry_threshold']):
                    if self.portfolio_manager.portfolio['cash'] >= stock2_data['Close/Last']:
                        self.portfolio_manager.buy_stock(stock2, stock2_data['Close/Last'])

                if normal_ratio * (1 - self.parameters['exit_threshold']) < current_ratio < normal_ratio * (1 + self.parameters['exit_threshold']):
                    # Logic to close the trade here
                    pass


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
                print(f"Debug: {stock} not found in current_prices. Available keys: {list(current_prices.keys())}")
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
            self.parameters['max_trade_quantity'] = int(self.parameters['max_trade_quantity'] * (1 + increment_factor))
        elif roi < last_roi:  # If ROI has decreased
            self.parameters['max_trade_quantity'] = max(1, int(self.parameters['max_trade_quantity'] * (1 - increment_factor)))
        # Additional logic for adjusting other strategy thresholds can be added here
            
# ... [Previous class definitions: DataProcessor, PairAnalyzer, PortfolioManager, TradingStrategy, PerformanceAnalytics] ...

# Initialize classes with the dataset file path
data_processor = DataProcessor('C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv')
pair_analyzer = PairAnalyzer(parameters)

# Initialize PortfolioManager once outside the generation loop
portfolio_manager = PortfolioManager(100000, parameters)
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
    last_roi = 0

    for trading_day in unique_trading_days:
        trading_day = pd.Timestamp(trading_day)
        print(f"Processing trading day: {trading_day.strftime('%Y-%m-%d')}")

        historical_data = data_processor.data[data_processor.data['Date'] <= trading_day]
        pivoted_historical_data = historical_data.pivot(index='Date', columns='Company', values='Close/Last')

        current_correlation_matrix = pivoted_historical_data.corr()
        current_pairs = pair_analyzer.identify_pairs(current_correlation_matrix)
        formatted_current_pairs = [PairAnalyzer.reformat_pair_names(pair) for pair in current_pairs]
        current_normal_ratios = {pair: pair_analyzer.calculate_normal_ratios(pivoted_historical_data, pair) for pair in formatted_current_pairs}

        daily_data = data_processor.data[data_processor.data['Date'] == trading_day]
        if daily_data.empty:
            continue

        # Update current_prices for the day
        current_prices = {row['Company'].strip(): row['Close/Last'] for _, row in daily_data.iterrows()}

        aggregated_day_data = {row['Company'].strip(): row for _, row in daily_data.iterrows()}
        for _, row in aggregated_day_data.items():
            trading_strategy.momentum_trading(row)
            trading_strategy.mean_reversion(row)
            trading_strategy.swing_trading(row)

        trading_strategy.pair_trading(aggregated_day_data, formatted_current_pairs, current_normal_ratios)

        current_value = performance_analytics.calculate_portfolio_value(portfolio_manager.portfolio, current_prices)
        portfolio_values.append(current_value)

        if len(portfolio_values) % evaluation_interval == 0:
            current_roi = performance_analytics.calculate_roi(100000, current_value)
            max_drawdown = performance_analytics.calculate_maximum_drawdown(portfolio_values[-evaluation_interval:])
            print(f"Day {trading_day.strftime('%Y-%m-%d')}: ROI: {current_roi:.2%}, Max Drawdown: {max_drawdown:.2%}")
            performance_analytics.adjust_parameters_based_on_roi(current_roi, last_roi)
            last_roi = current_roi

    # Just before the final valuation
    print(f"Final Portfolio before valuation: {portfolio_manager.portfolio}")
    print(f"Final Current Prices: {current_prices}")

    final_value = performance_analytics.calculate_portfolio_value(portfolio_manager.portfolio, current_prices)
    final_roi = performance_analytics.calculate_roi(100000, final_value)

    print(f"End of Generation {generation + 1}: Final ROI: {final_roi:.2%}")
    print(f"Total Portfolio Value at End of Generation {generation + 1}: ${final_value:,.2f}")
    print(f"Total Profit at End of Generation {generation + 1}: ${final_value - 100000:,.2f}")

    if generation == num_generations - 1:
        daily_portfolio_values_last_gen = portfolio_values.copy()

# After the loop over generations
final_portfolio = portfolio_manager.portfolio

# Extracting stock names and their quantities from the final portfolio
stock_names = list(final_portfolio['stocks'].keys())
quantities = [final_portfolio['stocks'][stock]['quantity'] for stock in stock_names]

# Plotting daily portfolio values for the last generation
plt.figure(figsize=(14, 7))
plt.plot(daily_portfolio_values_last_gen)
plt.title('Daily Portfolio Value')
plt.xlabel('Day')
plt.ylabel('Portfolio Value')
plt.grid(True)
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