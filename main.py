import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Monte Carlo Options Pricing")
st.sidebar.title("Simulation Parameters")

# Parameters for Monte Carlo simulation
ticker = st.sidebar.text_input('Ticker symbol', 'AAPL')
#option_type = st.sidebar.text_input('Types of Options', 'call or put')
option_type = st.sidebar.selectbox(
        'Types of Options', ("call", "put")
    )
strike_price = st.sidebar.number_input('Strike price', 0)
risk_free_rate = st.sidebar.slider('Risk-free rate (%)', 0, 100, 10)
sigma = st.sidebar.slider('Sigma (%)', 0, 100, 20)
exercise_date = st.sidebar.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
number_of_simulations = st.sidebar.slider('Number of simulations', 100, 100000, 10000)
num_of_movements = st.sidebar.slider('Number of price movement simulations to be visualized', 0, int(number_of_simulations / 10), 100)
st.sidebar.markdown(f"Credit: https://www.linkedin.com/in/tassatap-sanguansuk-b7b508237/")

# Calculate Monte Carlo Simulation of Stock Prices
def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps, option_type='call'):
    dt = T / num_steps  # Time increment
    simulations = np.zeros((num_steps + 1, num_simulations))  # Matrix to store simulated prices
    simulations[0] = S0  # Initial stock price

    # Simulate stock price paths
    for i in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)  # Random normal variables
        simulations[i] = simulations[i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return simulations

if st.button(f'Calculate option price for {ticker}'):
    # Show current stock data
    df = yf.download(ticker)
    st.write(df.tail())

    # Formatting simulation parameters
    spot_price = df['Close'][-1]
    risk_free_rate = risk_free_rate / 100
    sigma = sigma / 100
    days_to_maturity = (exercise_date - datetime.now().date()).days / 365.0  # Convert days to years

    monte_carlo = monte_carlo_option_pricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations, 252, option_type)
    S_T = monte_carlo[-1]
    
    # Calculate the payoff at maturity
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - strike_price, 0)  # Call option payoff
    elif option_type.lower() == 'put':
        payoffs = np.maximum(strike_price - S_T, 0)  # Put option payoff
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    sim_option_price = np.mean(payoffs)*np.exp(-risk_free_rate*days_to_maturity) #discounting back to present value

    # Plot the simulation results
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(num_of_movements, number_of_simulations)):
        ax.plot(monte_carlo[:, i], lw=0.5)
    
    ax.set_title("Monte Carlo Simulation of Stock Prices")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Stock Price")
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Displaying call/put option price
    st.subheader(f'simulated option price: {sim_option_price}')
    
    
