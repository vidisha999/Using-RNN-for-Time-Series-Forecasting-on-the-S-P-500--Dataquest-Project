# Time-Series-Forecasting-on-the-S&P500-using a LSTM model Dataquest-Project

## Project Description 
The focus of this project is to build a time-series forecasting model using LSTM(Long Sshort Term Memory) which is a type of RNN ( Recurrent Neural Network) to predict the move of the S&P index in the future based on its behavior over the past years.

## Background
S&P index is a stock market index tracking the performance of the 500 largest publicly traded companies listed in the stock exchnage of the USA. Tracking the S&P index is important because it acts as a market performance gauge, benchmark for investment strategies and diversification of the company with reducing portfolio risk by getting exposed to wide range of companies. With better and accurate predictions,business can have a great understanding on how to trade their stocks in the future.

## Data
[Yahoo Finance via Kaggle](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price) dataset which contains information about S&P 500 index from 2015 to 2020 is used in this project. 
The data has several potential columns that can be use for forecasting of the S&P index. 
- `Open`: The openinig value of the index on that day
- `High`: The highest value of the index on that day 
- `Low` : The lowest value of the index on that day 
- `Close`: The closing value of the index on that day 
- `Volume`: The total volume of the index traded on that day 
- `Adj Close`: The closing value of the index on that day adjusted for dividends.

Since the `Adj Close` which is the adjusted closing price on that day takes the dividends, stock splits, and new stock offerings into account, it is more appropriate to use for forecasting purposes as it's a more accurate representation of the stock value by being accounted for discrete jumps in index due to paying out dividends.

## Exploratory Data Analysis (EDA) 

During the data wrangling process, only `Adj Close` and ` Date` columns from the dataset were extracted and `Date` column was set as the index of the dataset, in order to prepare the data for the time series forecasting.

The following insights were gained from the exploration of the data: 

- There were no any missing vlaues for the dataset, which makes it appropriate for the data modelling.
- The mean value of the adjusted closing price of the stock is $2647.85
- The skewness of the dataset gained for the variable `Adj Close` is 0.08 which is a really small value indicating the data distribution is relatively symmetrical.
- By plotting the change of S&P index over yearly basis as shown in [S&P index Vs.Date plot](Images/RNN_fig1.png), an overall  pattern with an increasing trend in the value of index was observed.

## Data Preprocessing 

  When splitting the cleaned dataset into training,validation and testing splits, the continuous stretches of the dataset is used  instead of randomly splits,since the dataset is used for time series prediction. 

The below code shows the split sizes for each dataset:
```python
train_size=int(len(stock_data)*0.5)
validation_size=int(len(stock_data)*0.25)
test_size=int(len(stock_data)*0.25)
```
Then the training split was fit to the MinMax scaler and all three datasets were transformed to x,y variables using the window technique as shown in the below code : 

``` python
def create_dataset(dataset,window_size=1):
    data_x,data_y=[],[]
    for i in range (len(dataset)-window_size-1):
            window=dataset.iloc[i:(i+window_size),0]
            target=dataset.iloc[i+window_size,0]
            data_x.append(window)
            data_y.append(target)
    return np.array(data_x),np.array(data_y)
```
Finally the x,y variables of each dataset is reshaped to a numpy array with a single timestep as the required format to use in the LSTM model.          
Similar to how the X_train dataset is transformed to a numpy array by using the below code, the x components of the validation and testing datasets were reshaped to numpy arrays. 
```python
X_train=np.reshape(X_train,X_train.shape[0],1,x_train.shape[1])
```
## Model Building 

### LSTM model 

The LSTM model was built as indicated in the below code: 
```python
model=tf.keras.Sequential()
model.add(layers.LSTM(10,input_shape=(1,window_size),activation='relu'))
model.add(layers.Dense(10,activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

# Training the model 
model.fit(X_train,y_train) # Fit the model
y_pred=model.predict(X_validation)
print(f'model\'s R2 score is : {r2_score(y_validation,y_pred)}')
```
### Modified LSTM model with a convolutional layer 

``` python
model=tf.keras.Sequential()
model.add(layers.Conv1D(64,1,input_shape=(1,window_size),activation='relu'))
model.add(layers.MaxPooling1D(1))
model.add(layers.LSTM(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

#Fit and predict the accuracy of the model 
model.fit(X_train,y_train) # Fit the model
y_pred=model.predict(X_validation)
print(f'model\'s R2 score is : {r2_score(y_validation,y_pred)}')
```
 ### Model Optimization 
 The above model was further optimized : 
 1. By setting the window size of the time-series data to a higher value of 25, to obtain a smoother statistic by capturing longer-term trends while being less responsive to short-term fluctuations in the data.
 2. By adding more dense connected hidden layers included with larger number of neurons in each layer, to allow model to learn more complpex patterns and representations of the data. 
## Model evaluation 

The co-efficient of determination (R2 score) is used to measure the model's performance. It is a statistical measure that indicates how well a model explains the variance in the observed data, essentially showing how much of the variation in the target variable can be attributed to the model's predictions. With time series data, causation should be used with the R2 score to minimize the limitation to non-constant and time-dependent data. 

- The negative R2 score of -92.97 obtained from the simple RNN model indicates a very poor model fit. 
- The R2 score of -60.2 obtained from LSTM model had been improved to -23.8, when the LSTM model was modified by adding a convolutional layer.
- After running the optimized model for 35 epochs, R2 score of 0.95 was obtained for the validation dataset, indicating a strong fit of the data to the model.
- The R2 score obtained for the test data on the optimized model, was 0.93, which is still a great value showing a well performacne of the model.

The performance of the model is visualized, by the predictions gained from  training, validation and testing datasets using the trained model and comparing it against the variability of the S&P index using the original dataset as shown in the [visualization of performance plot](Images/RNN_fig2.png).

## Conclusion 
Based on the visualization plot, the model's predictions allign with the observed values of the original dataset and it confirms the higher value obtained for the statistical meassure ( R2 score) indicating a strong fit of data to the optimized model. So, this model can be used for time-series forecasting of the variability of S&P index of future stock values, offering insights on the overall health of the US stock  markets,leading a less risky investment decision making  process. 




















































