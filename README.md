# Stock Price Forecast Using Online Learning Model

## Introduction

Continuous variation in stock market lays people into confusion. Forecasting stock price in real time imposes a challenge for human as it is rather difficult to find the underlying patterns or relationships in the massive amount of data when the data coming in are continuous and ever-changing. This project aims to develop a robust predictive model to forecast the stock market trend using real-time financial time series data, by extending Online Recurrent Extreme Learning Machine (OR-ELM) algorithm, and introducing two optimization strategies: adaptive forgetting factor (AFF) and Genetic Algorithm (GA). A comparative study is carried out to compare the forecasting performance of existing algorithms with the implemented algorithm (ORELM-AFF-GA). 

## Dataset
**Nikkei 225**
- A stock market index for Tokyo Stock Exchange (TSE)
- Intra-day historical data obtained from [Yahoo Finance](https://finance.yahoo.com/)
- Period: 1/3/1966 to 29/2/2018

## Algorithms
1. **Online Sequential Extreme Learning Machine (OS-ELM)**  
- An online learning method based on Extreme Learning Machine (ELM)  
Paper: *Liang, Nan-Ying, et al. "A fast and accurate online sequential learning algorithm for feedforward networks." IEEE Transactions on neural networks 17.6 (2006): 1411-1423.*

<p> &nbsp</p>

2. **Online Recurrent Extreme Learning Machine (OR-ELM)**   
- Paper: *Jin-Man Park, and Jong-Hwan Kim. "Online recurrent extreme learning machine and its application to time-series prediction." Neural Networks (IJCNN), 2017 International Joint Conference on. IEEE, 2017.*  

- Based on Fully Online Sequential Extreme Learning Machine (FOS-ELM)  
*Wong, Pak Kin, et al. "Adaptive control using fully online sequential-extreme learning machine and a case study on engine air-fuel ratio regulation." Mathematical Problems in Engineering 2014 (2014).*

- FOS-ELM + Layer Normalization + forgetting factor + weight auto-encoding (input->hidden, hidden->hidden)

<p> &nbsp</p>

3. **Online Recurrent Extreme Learning Machine with Adaptive Forgetting Factor (ORELM-AFF) (PROPOSED)**
- Integrates OR-ELM with adaptive forgetting factor (FF) to handle time-varying sequential data, since a constant forgetting factor may not be sufficient to track all the system dynamics
- With this forgetting mechanism, the forgetting factor, λ will be adjusted based on the changes of the data characteristics.
- Specifically, when there is a abrupt change in the data distribution, a smaller forgetting factor will be used to adapt to the trend and discard the older data that will bring negative effect on the prediction accuracy. On the contrary, a larger forgetting factor will be used when the trend is steadier to increase the memory length of the algorithm. 
- The recursive update of the forgetting factor was adopted from the research conducted by Li, Zhang, Yin, Xiao & Zhang (2017).

<p align="center">
  <img src ="https://s3-ap-southeast-1.amazonaws.com/mhlee2907/time+series+7.jpg" />
</p>

4. **Online Recurrent Extreme Learning Machine with Adaptive Forgetting Factor and Genetic Algorithm (ORELM-AFF-GA) (PROPOSED)**
- Added Steady State Genetic Algorithm Genetic Algorithm (GA) for hyperparamater optimization.
- Efficiently select optimal parameters (window size and number of hidden neurons) in a large complex search space.

 Below figure depicts a 3-dimensional scatter plot showing the distribution of the solutions generated using GA where a lighter-colored point indicates better solution sets and darker-colored point represents weaker solution sets.

<p align="center">
  <img src ="https://s3-ap-southeast-1.amazonaws.com/mhlee2907/time+series+3.JPG"  height="350" width="400" />
</p>

## Conceptual Framework
<p align="center">
  <img src ="https://s3-ap-southeast-1.amazonaws.com/mhlee2907/time+series+8.JPG"  height="450" width="400" />
</p>


## Requirements 
- Python 2.7
- Expsuite (included in this repository)

## Usage

Run prediction code:

    python run.py 
    
Use *-a* to run other algorithms (OSELM or ORELM). For example:

    python run.py -a OSELM
    
## Result
**ORELM-AFF-GA** shows superior performance in capturing the non-linearity and predicting the stock price.

<p align="center">
  <img src ="https://s3-ap-southeast-1.amazonaws.com/mhlee2907/time+series+4.JPG"  height="370" width="550" />
</p>

Prediction Performance of all algorithms:

<p align="center">
  <img src ="https://s3-ap-southeast-1.amazonaws.com/mhlee2907/time+series+2.JPG"  height="500" width="600" />
</p>


## Rerefences
Jin-Man Park, and Jong-Hwan Kim, Online-Recurrent-Extreme-Learning-Machine, (2017), GitHub repository, https://github.com/chickenbestlover/Online-Recurrent-Extreme-Learning-Machine

Li, Y., Zhang, S., Yin, Y., Xiao, W., & Zhang, J. (2017). A Novel Online Sequential Extreme Learning Machine for Gas Utilization Ratio Prediction in Blast Furnaces. Sensors, 17(8), 1847. doi: 10.3390/s17081847
