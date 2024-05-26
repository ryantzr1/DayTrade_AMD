# DayTrade_AMD
LSTM model for predicting day trading signals with attention mechanism and multi-source data integration.

![GitHub stars](https://img.shields.io/github/stars/ryantzr1/DayTrade_AMD?style=social)

Welcome to the **DayTrade_AMD** project! This repository contains a sophisticated machine learning model designed to predict day trading decisions for AMD stocks. Leveraging a combination of historical daily stock data, 15-minute interval data, historical news data, macroeconomic indicators, and competitors' news, this model aims to provide high-accuracy predictions for traders. Although initially designed for AMD, this model can be adapted to work with any stock.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Description
**DayTrade_AMD** is a machine learning model tailored for day traders looking to make informed decisions on AMD stocks. The model employs an LSTM network with an attention mechanism to analyze a variety of features and output a probability score indicating whether to buy the stock. The framework is versatile and can be adapted to predict trading signals for any stock by feeding it the appropriate data.

## Features
- Predictive model for day trading decisions
- Utilizes historical daily stock data and 15-minute interval data
- Incorporates historical news data from GDELT
- Considers macroeconomic indicators and competitors' news
- Implements an LSTM model with an attention mechanism
- Adaptable to any stock with relevant data

## Installation
To get started with **DayTrade_AMD**, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/ryantzr1/DayTrade_AMD.git
   ```
2. Navigate to the project directory:
    ```sh
    cd DayTrade_AMD
    ```

## Usage
To use the DayTrade_AMD model, simply run the code provided in the repository. If you wish to predict trading signals for a different stock, change the stock symbol from "AMD" to your desired stock symbol. Then, run the function and check the results. If the score is above 0.5, you can consider buying the stock.

To change the stock symbol, edit the following line in the code:
```sh
symbol = "AMD"
```

To see the prediction results, run this function:
```sh
def predict_next_2_hours_profit(shares, model):
```
This function takes the number of shares and the trained model as inputs. It fetches the most recent stock data, processes it, and predicts whether the stock is likely to be profitable within the next two hours. If the prediction score is above 0.5, it suggests entering the trade.



## Model Architecture
The DayTrade_AMD model is built using a Long Short-Term Memory (LSTM) network. Here’s a detailed breakdown of the architecture:

- Input Layer: Accepts sequences of features, where each sequence is composed of multiple time steps. This layer processes different types of data to form a comprehensive input vector.
- LSTM Layer: This layer is crucial for capturing temporal dependencies in the time-series data. The LSTM network helps in understanding the sequential nature of stock prices and related features. It has 64 units and uses the tanh activation function.
- Dropout Layer: This layer helps prevent overfitting by randomly dropping 25% of the neurons during training. It improves the model's ability to generalize to new data.
- Dense Layer: This fully connected layer performs the final transformations and produces the output probability. It combines the features extracted by the LSTM layer to form a coherent prediction.
- Output Layer: The final layer is a single neuron with a sigmoid activation function, outputting the probability of a BUY signal. If the probability is above 0.5, the recommendation is to BUY, otherwise to HOLD/SELL.

## Technical Insights and Inspirations
The design and implementation of the DayTrade_AMD model draw inspiration from the Stanford University report "Using LSTM in Stock Prediction and Quantitative Trading" by Zhichao Zou and Zihao Qu. This paper highlights the effectiveness of LSTM networks, particularly Attention-LSTM, in capturing temporal dependencies and patterns in stock price data, making them suitable for time-series forecasting tasks.

Key takeaways from the paper include:
- Handling Temporal Dependencies: LSTM networks are adept at capturing long-term dependencies in sequential data, essential for modeling stock prices influenced by past events.
- Feature Engineering: Incorporating various features such as historical prices, trading volumes, and corporate accounting statistics (e.g., Debt-to-Equity Ratio, Return on Equity) can enhance model performance.
- Attention Mechanism: The paper discusses the use of attention mechanisms to improve LSTM performance by allowing the model to focus on the most relevant parts of the input sequence. This helps in better capturing the impact of significant events on stock prices.
- Model Evaluation: Evaluating model performance using metrics such as Mean Squared Error (MSE) and comparing trading strategies helps in fine-tuning the architecture and improving prediction reliability.
By integrating these insights, the DayTrade_AMD model aims to provide robust and reliable trading signals for day traders. You can read more about the research and technical details in the paper [here](https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf).


## Future Work
In the pipeline for DayTrade_AMD are enhancements to further improve prediction accuracy:
- Integration of Additional Data Sources: Incorporating more news sources and macroeconomic data to capture a broader range of factors affecting stock prices.
- Sentiment Analysis: Incorporating sentiment analysis of news articles to better gauge market sentiment and its impact on stock prices.
- Attention Layer: Adding an attention layer when more data is available to improve the model's ability to focus on the most relevant information.
- Exploring Alternative Models: Testing other model architectures, such as GRU (Gated Recurrent Unit), and performing hyperparameter tuning to optimize performance.

## Contributing
We welcome contributions to the DayTrade_AMD project! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
