import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import itertools


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
    "exit_threshold": 0.03,
    "max_trade_quantity": 10,
    "historical_window": 60,
    "correlation_threshold": 0.95,
    "evaluation_interval": 30
    # "increment_factor": 0.01
}

parameter_space = {
    "momentum_threshold": [0.025, 0.05, 0.075],
    "mean_rev_threshold": [0.025, 0.05, 0.075],
    "swing_threshold": [0.025, 0.05, 0.075],
    "entry_threshold": [0.025, 0.05, 0.075],
    "exit_threshold": [0.025, 0.05, 0.075],
    "max_trade_quantity": [10, 15, 20],
    "historical_window": [60],
    "correlation_threshold": [0.85, 0.90, 0.95],
    "evaluation_interval": [30]
}


class PairAnalyzer:
    def __init__(self, parameters):
        self.parameters = parameters

    def identify_pairs(self, correlation_matrix):
        pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.parameters['correlation_threshold']:
                    pairs.append(
                        (correlation_matrix.columns[i], correlation_matrix.columns[j]))
        return pairs

    def calculate_normal_ratios(self, pivoted_data, pair):
        # Convert historical_window to an integer to prevent type issues
        historical_window = int(self.parameters['historical_window'])
        daily_ratios = pivoted_data[pair[0]] / pivoted_data[pair[1]]
        return daily_ratios[-historical_window:].mean()

    @staticmethod
    def reformat_pair_names(pair):
        return tuple(re.sub(r'\d+\.', '', stock).strip() for stock in pair)


class PortfolioManager:
    def __init__(self, initial_capital, parameters):
        self.portfolio = self.initialize_portfolio(initial_capital)
        self.parameters = parameters

    def initialize_portfolio(self, initial_capital):
        return {'cash': initial_capital, 'stocks': {}}

    def update_parameters(self, new_parameters):
        """Update trading parameters with new values."""
        self.parameters.update(new_parameters)

    def buy_stock(self, stock, price):
        quantity = min(
            self.parameters['max_trade_quantity'], self.portfolio['cash'] // price)
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

            print(f"After buying {stock}, portfolio: {self.portfolio}")

    def sell_stock(self, stock, price):
        if stock in self.portfolio['stocks']:
            quantity = min(
                self.parameters['max_trade_quantity'], self.portfolio['stocks'][stock]['quantity'])
            if quantity > 0:
                self.portfolio['cash'] += price * quantity
                self.portfolio['stocks'][stock]['quantity'] -= quantity

                if self.portfolio['stocks'][stock]['quantity'] == 0:
                    del self.portfolio['stocks'][stock]

                print(f"After selling {stock}, portfolio: {self.portfolio}")

    # Additional methods for portfolio management can be added here, if needed.


class TradingStrategy:
    def __init__(self, portfolio_manager, parameters):
        self.portfolio_manager = portfolio_manager
        self.parameters = parameters

    def momentum_trading(self, company, price, prediction):
        if prediction > price * (1 + self.parameters['momentum_threshold']):
            self.portfolio_manager.buy_stock(company, price)
        elif prediction < price * (1 - self.parameters['momentum_threshold']):
            self.portfolio_manager.sell_stock(company, price)

    def mean_reversion(self, company, price, thirty_day_ma):
        if price < thirty_day_ma * (1 - self.parameters['mean_rev_threshold']):
            self.portfolio_manager.buy_stock(company, price)
        elif price > thirty_day_ma * (1 + self.parameters['mean_rev_threshold']):
            self.portfolio_manager.sell_stock(company, price)

    def swing_trading(self, company, price, prediction):
        if prediction > price * (1 + self.parameters['swing_threshold']):
            self.portfolio_manager.buy_stock(company, price)
        elif prediction < price * (1 - self.parameters['swing_threshold']):
            self.portfolio_manager.sell_stock(company, price)

    def pair_trading(self, aggregated_day_data, pairs, normal_ratios):
        for pair in pairs:
            stock1, stock2 = pair

            if stock1 in aggregated_day_data and stock2 in aggregated_day_data:
                stock1_data = aggregated_day_data[stock1]
                stock2_data = aggregated_day_data[stock2]
                current_ratio = stock1_data['Close/Last'] / \
                    stock2_data['Close/Last']
                normal_ratio = normal_ratios[pair]

                if current_ratio < normal_ratio * (1 - self.parameters['entry_threshold']):
                    if self.portfolio_manager.portfolio['cash'] >= stock1_data['Close/Last']:
                        self.portfolio_manager.buy_stock(
                            stock1, stock1_data['Close/Last'])

                elif current_ratio > normal_ratio * (1 + self.parameters['entry_threshold']):
                    if self.portfolio_manager.portfolio['cash'] >= stock2_data['Close/Last']:
                        self.portfolio_manager.buy_stock(
                            stock2, stock2_data['Close/Last'])

                if normal_ratio * (1 - self.parameters['exit_threshold']) < current_ratio < normal_ratio * (1 + self.parameters['exit_threshold']):
                    # Logic to close the trade here
                    pass


class PerformanceAnalytics:
    def __init__(self, parameters):
        self.parameters = parameters
        self.parameter_space = parameter_space

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

    def select_parameters(self, opening_price, previous_closing_price):
        if opening_price > previous_closing_price * 1.01:
            return "upper_bound"
        elif opening_price < previous_closing_price * 0.99:
            return "lower_bound"
        else:
            return "original"

    def simulate_trading(self, selected_parameters, daily_data, current_prices, opening_price, previous_closing_price):
        # Temporarily store original parameters
        original_parameters = self.parameters.copy()

        # Select the type of predicted price based on opening and previous closing prices
        price_type = self.select_parameters(
            opening_price, previous_closing_price)

        # Run the trading strategies for the day
        for _, row in daily_data.iterrows():
            company = row['Company'].strip()
            price = row['Close/Last']
            # Select the appropriate predicted price based on the price type
            if price_type == "upper_bound":
                prediction = row.get(
                    'Prediction Upper Bound', row['Prediction'])
            elif price_type == "lower_bound":
                prediction = row.get(
                    'Prediction Lower Bound', row['Prediction'])
            else:  # "original"
                prediction = row['Prediction']

            # Default to 0 if 30d MA is not available
            thirty_day_ma = row.get('30d MA', 0)

            # Update parameters for the trading strategy
            trading_strategy.update_parameters(selected_parameters)

            trading_strategy.momentum_trading(company, price, prediction)
            trading_strategy.mean_reversion(company, price, thirty_day_ma)
            trading_strategy.swing_trading(company, price, prediction)

        # Calculate the portfolio value after the day's trading
        portfolio_value = self.calculate_portfolio_value(
            portfolio_manager.portfolio, current_prices)
        daily_roi = self.calculate_roi(50000, portfolio_value)

        # Reset parameters to original after simulation
        self.parameters = original_parameters

        return daily_roi, portfolio_value

    def evaluate_trading_strategy(self, predicted_price_type, parameter_combinations, daily_data, current_prices):
        best_portfolio_value = float('-inf')
        best_parameters = None

        for params_tuple in parameter_combinations:
            # Convert tuple back to dictionary
            selected_parameters = dict(
                zip(parameter_space.keys(), params_tuple))

            # Simulate trading with the current set of parameters
            _, portfolio_value = self.simulate_trading(
                selected_parameters, daily_data, current_prices, predicted_price_type)

            # Check if the current set of parameters results in the highest portfolio value so far
            if portfolio_value > best_portfolio_value:
                best_portfolio_value = portfolio_value
                best_parameters = selected_parameters

        return best_parameters, best_portfolio_value


# Initialize classes with the dataset file path
data_processor = DataProcessor(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv')
pair_analyzer = PairAnalyzer(parameters)

# Initialize PortfolioManager and TradingStrategy
portfolio_manager = PortfolioManager(50000, parameters)
trading_strategy = TradingStrategy(portfolio_manager, parameters)

# Generate all possible combinations of parameters
parameter_combinations = list(itertools.product(
    *(parameter_space[key] for key in sorted(parameter_space))))

# Initialize PerformanceAnalytics with parameters
performance_analytics = PerformanceAnalytics(parameters)

# Prepare for the simulation
daily_portfolio_values_last_gen = []
num_generations = 1

unique_trading_days = pd.to_datetime(data_processor.data['Date'].unique())
unique_trading_days = np.sort(unique_trading_days)

for generation in range(num_generations):
    print(f"\nStarting Generation {generation + 1}")
    portfolio_values = []

    for trading_day in unique_trading_days:
        trading_day = pd.Timestamp(trading_day)
        print(f"Processing trading day: {trading_day.strftime('%Y-%m-%d')}")

        # Prepare data for the day's trading
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

        # Extract opening price and previous closing price
        opening_price = daily_data['Open'].iloc[0]
        # Handle the case for the first day in the dataset
        previous_day_data = data_processor.data[data_processor.data['Date'] < trading_day]
        if previous_day_data.empty:
            previous_closing_price = opening_price  # Use opening price as a fallback
        else:
            previous_closing_price = previous_day_data['Close/Last'].iloc[-1]

        current_prices = {row['Company'].strip(): row['Close/Last']
                          for _, row in daily_data.iterrows()}

        # Evaluate trading strategy for the day
        best_parameters, _ = performance_analytics.evaluate_trading_strategy(
            parameter_combinations, daily_data, current_prices, opening_price, previous_closing_price)

        # Apply the best parameters and execute trading strategies
        portfolio_manager.update_parameters(best_parameters)
        for _, row in daily_data.iterrows():
            company = row['Company'].strip()
            price = row['Close/Last']
            prediction = row['Prediction']
            thirty_day_ma = row['30d MA']

            trading_strategy.momentum_trading(company, price, prediction)
            trading_strategy.mean_reversion(company, price, thirty_day_ma)
            trading_strategy.swing_trading(company, price, prediction)

        # Calculate and store the day's portfolio value
        current_value = performance_analytics.calculate_portfolio_value(
            portfolio_manager.portfolio, current_prices)
        portfolio_values.append(current_value)

        # Calculate ROI for the day
        daily_roi = performance_analytics.calculate_roi(50000, current_value)

        # Print portfolio value and ROI
        print(
            f"End of {trading_day.strftime('%Y-%m-%d')}: Portfolio Value: ${current_value:,.2f}, ROI: {daily_roi:.2%}")

    # End of generation processing
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

# Extracting final portfolio details
final_portfolio = portfolio_manager.portfolio
stock_names = list(final_portfolio['stocks'].keys())
quantities = [final_portfolio['stocks'][stock]['quantity']
              for stock in stock_names]


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

