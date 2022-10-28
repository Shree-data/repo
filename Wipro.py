#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install nsepy')


# In[2]:


#from nsepy import get_history
from datetime import date
import pandas as pd


# In[3]:


start_date_W = date(2012, 10, 1)
end_date_W = date(2022,10,15)

Wipro = pd.read_csv('F:\\Project\\Wipro.csv')
print(Wipro)


# In[4]:


stock_W = pd.DataFrame(Wipro)
stock_W


# In[181]:


stock_W.to_csv('Wipro.csv')




# ### EDA on Wipro Data

# In[9]:


stock_W


# In[10]:


stock_W.shape


# In[11]:


stock_W.info()


# In[12]:


stock_W.reset_index(inplace = True)
stock_W.info()


# In[13]:


stock_W['Date'] = pd.to_datetime(stock_W['Date'])
stock_W.info()


# In[14]:


stock_W.set_index('Date', inplace = True)
stock_W.info()


# In[15]:


stock_W.describe()


# In[16]:


stock_W.isnull().sum()


# In[17]:


W = stock_W.drop(columns =['Symbol', 'Series','VWAP','Volume','Turnover', 'Trades', 'Deliverable Volume', '%Deliverble'])
W


# In[18]:


W.corr()


# In[19]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')


# In[20]:


close_W = pd.DataFrame(W['Close'])
close_W


# In[21]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(close_W)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Prise', fontsize = 18)
plt.show()


# In[22]:


#moving average
ma100 = W.Close.rolling(100).mean()
ma100


# In[23]:


plt.figure(figsize=(12,6))
plt.plot(W.Close)
plt.plot(ma100, 'r')


# In[24]:


plt.hist(close_W, bins = 10);


# In[25]:


import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import lag_plot


# In[26]:


sns.boxplot(W['Close'])


# In[27]:


W['Close'].plot(kind = 'kde')


# In[28]:


lag_plot(close_W)
pyplot.show()


# In[29]:


plt.figure(figsize=(12,6))
plt.plot(W.Close)
plt.plot(W.Open, 'r')


# In[30]:


sns.heatmap(W.corr(), cmap = "YlGnBu")


# In[31]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(close_W, model='multiplicative', period = 365)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)


# In[32]:


w_cumsum = W.cumsum()
w_cumsum.plot()
plt.title('Wipro Cumulative Returns')


# In[33]:


w_cumsum = close_W.cumsum()
w_cumsum.plot()
plt.title('Wipro Cumulative Returns')


# In[34]:


stock_W['Close'].resample(rule = 'M').mean().plot(figsize=(15,10))


# ## Modelling

# In[103]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(close_W.squeeze(), lags = 20)
plot_acf(close_W.squeeze(), lags = 20)


# ## Splitting Data into Train and Test

# In[104]:


split_point = len(close_W) - 365
train_W = close_W[0:split_point]
test_W = close_W[split_point:]


# In[105]:


train_W.shape


# In[106]:


test_W.shape


# ## Naive Bayes

# In[107]:


sp =len(stock_W) - 30
train_W1= W[0:sp]
test_W1 = W[sp:]


# In[108]:


y_hat = test_W.copy() 
y_hat['naive'] = test_W 
plt.figure(figsize=(12,8)) 
plt.plot(train_W.index, train_W['Close'], label='Train') 
plt.plot(test_W.index,test_W['Close'], label='Valid') 
#plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title("Naive Forecast") 
plt.show()


# In[109]:


y_hat = test_W.copy() 
y_hat['naive'] = test_W 
plt.figure(figsize=(12,8)) 
plt.plot(train_W.index, train_W['Close'], label='Train') 
plt.plot(test_W.index,test_W['Close'], label='Valid') 
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title("Naive Forecast") 
plt.show()


# In[110]:


from sklearn.metrics import mean_squared_error 
from math import sqrt 
rms = sqrt(mean_squared_error(test_W.Close, y_hat.naive)) 
print(rms)


# In[111]:


colnames = W.columns


# In[112]:


#2

x_train = train_W1[colnames[0:5]].values
y_train = train_W1[colnames[5]].values
x_test = test_W1[colnames[0:5]].values
y_test = test_W1[colnames[5]].values


# In[113]:


def norm_func(i):
    x = (i-i.min())/(i.max() - i.min())
    return (x)


# In[114]:


x_train = norm_func(x_train)
x_test = norm_func(x_test)


# In[115]:


from sklearn.naive_bayes import MultinomialNB as MB

M_model = MB()
train_pred_multi = M_model.fit(x_train, y_train.astype('int')).predict(x_train)
test_pred_multi = M_model.fit(x_test, y_test.astype('int')).predict(x_test)


# In[116]:


train_acc_multi = np.mean(train_pred_multi == y_train)
train_acc_multi


# In[117]:


test_acc_multi = np.mean(test_pred_multi== y_test)
test_acc_multi


# In[118]:


from sklearn.naive_bayes import GaussianNB as GB
G_model=GB()
train_pred_gau=G_model.fit(x_train,y_train.astype('int')).predict(x_train)
test_pred_gau=G_model.fit(x_test,y_test.astype('int')).predict(x_test)


# In[119]:


train_acc_gau = np.mean(train_pred_gau == y_train)
train_acc_gau


# In[120]:


test_acc_gau = np.mean(test_pred_gau == y_test)
test_acc_gau


# In[ ]:





# ## ARIMA

# In[121]:


train_data, test_data = stock_W[0:int(len(stock_W)*0.6)], stock_W[int(len(stock_W)*0.6):]
plt.figure(figsize=(12,7))
plt.title('Wipro Prices')
plt.xlabel('Dates')
plt.ylabel('Close')
plt.plot(stock_W['Close'], 'blue', label='Training Data')
plt.plot(test_data['Close'], 'green', label='Testing Data')
plt.legend()


# In[122]:


def wipro_val(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))


# ### Augumented Dickey - Fuller test
# 
# #### Null hypothesis - The series is stationary
# #### Alternate hypothesis - The series is not stationary

# In[123]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(close_W['Close'])
print('ADF Statistics: %f' %result[0])
print('p-value:%f' %result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' %(key,value))


# ### Here we see that p - value is more than 0.05. Hence our null hypothesis will be rejected . Thus this series is non-stationary.

# In[124]:


close_W.plot()


# #### Finding d - parameter

# In[125]:


plt.rcParams.update({'figure.figsize':(20,15),'figure.dpi':120})


# In[126]:


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(close_W.Close)
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)

#1st Differencing
ax2.plot(close_W.Close.diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)

#2nd Differencing
ax3.plot(close_W.Close.diff().diff())
ax3.set_title('2nd Order Differencing')
ax3.axes.xaxis.set_visible(False)

plt.show()


# #### Since comparitively, second order differencing has lesser noice than first order, we consider d = 2.

# In[127]:


from statsmodels.graphics.tsaplots import plot_acf
fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(close_W.Close, ax=ax1)
plot_acf(close_W.Close.diff().dropna(), ax=ax2)
plot_acf(close_W.Close.diff().diff().dropna(), ax=ax3)


# In[128]:


from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize = (20,15))
plot_pacf(close_W.Close.diff().diff().dropna())


# #### p = 1

# In[129]:


plot_acf(close_W.Close.diff().diff().dropna())


# #### q= 1

# In[130]:


from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as smapi
model = smapi.tsa.arima.ARIMA(close_W.Close, order = (1,2,1))
model_fit = model.fit()
model_fit.summary()


# In[131]:


train_w = train_data['Close'].values
test_w = test_data['Close'].values

history = [x for x in train_w]
print(type(history))
predictions = list()
for t in range(len(test_w)):
    model = ARIMA(history, order=(1,2,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_w[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))


# In[132]:


error = mean_squared_error(test_w, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = wipro_val(test_w, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)


# In[133]:


plt.figure(figsize=(12,7))
plt.plot(close_W['Close'], 'green', color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(test_data.index, test_data['Close'], color='red', label='Actual Price')
plt.title('Close price Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()


# In[134]:


pred_new = pd.DataFrame(model_fit.predict(start=0,end=6)[1:], columns = ['Prediction'])


# In[135]:


pred_new = pred_new.drop([0, 1])
pred_new.reset_index(inplace = True)
#pred_new.drop(columns =['index'])


# In[136]:


start_date_n = date(2022, 10, 15)
end_date_n = date(2022,10,21)

Wipro_n = get_history(symbol = 'WIPRO', start = start_date_n, end = end_date_n)
print(Wipro_n)


# In[137]:


w_n = pd.DataFrame(Wipro_n)


# In[138]:


w_nc = pd.DataFrame(w_n['Close'])
w_nc.reset_index(inplace=True)


# In[139]:


p_n = pd.concat([pred_new, w_nc], axis = 1)
p_n.drop(columns = ['index'])


# In[140]:


p_n.plot()


# ## LSTM

# In[141]:


from sklearn.preprocessing import MinMaxScaler


# In[142]:


close_w = close_W
close_w


# In[143]:


train_w = close_w.iloc[:2300]
test_w = close_w.iloc[2300:]


# In[144]:


test_w = test_w[0:184]


# In[145]:


test_w.info()


# In[146]:


from sklearn.preprocessing import MinMaxScaler


# In[147]:


scaler = MinMaxScaler()


# In[148]:


scaler.fit(train_w)


# In[149]:


scaler_train = scaler.transform(train_w)
scaler_test = scaler.transform(test_w)


# In[150]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[151]:


len(scaler_train)


# In[152]:


from keras.models import Sequential
from keras.layers import Dense,LSTM


# In[153]:


n_input = 184
n_feature = 1

train_generator_w = TimeseriesGenerator(scaler_train, scaler_train, length = n_input, batch_size = 1)


# In[154]:


model = Sequential()

model.add(LSTM(128, activation = 'relu', input_shape=(n_input, n_feature), return_sequences = True))
model.add(LSTM(128, activation = 'relu', input_shape=(n_input, n_feature), return_sequences = True))
model.add(LSTM(128, activation = 'relu', input_shape=(n_input, n_feature), return_sequences = False))
model.add(Dense(1))


# In[155]:


model.summary()


# In[156]:


model.compile(optimizer = 'adam', loss= 'mean_squared_error')
model.fit_generator(train_generator_w, epochs = 5)


# In[157]:


losses_lstm = model.history.history['loss']
plt.figure(figsize = (12,4))
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)), losses_lstm)


# In[158]:


lstm_predictions_scaled = list()

batch = scaler_train[-n_input:]
current_batch = batch.reshape((1, n_input, n_feature))

for i in range(len(test_w)):   
    lstm_pred = model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[159]:


lstm_predictions_scaled


# In[160]:


lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions


# In[161]:


test_w['LSTM_Predictions'] = lstm_predictions
test_w


# In[162]:


test_w['Close'].plot(figsize=(20,10), legend = True)
test_w['LSTM_Predictions'].plot(legend = True)


# In[163]:


from statsmodels.tools.eval_measures import rmse
lstm_rmse_error = rmse(test_w['Close'], test_w["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = close_W['Close'].mean()

print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}')


# ## Prophet

# In[164]:


close_W.info()


# In[165]:


w_pr = close_W.copy()


# In[166]:


w_pr = close_W.reset_index()


# In[167]:


w_pr.columns = ['ds', 'y']


# In[168]:


train_w_pr = w_pr.iloc[:len(close_W)-30]
test_w_pr = w_pr.iloc[len(close_W)-30:]


# In[169]:


test_w_pr


# In[170]:


get_ipython().system('pip install prophet')
from prophet import Prophet


# In[171]:


df_w = Prophet()
df_w.fit(train_w_pr) 
future = df_w.make_future_dataframe(periods = 30, freq='MS')
prophet_pred_w = df_w.predict(future)


# In[172]:


prophet_pred_w


# In[173]:


prophet_pred_w = pd.DataFrame({'Date': prophet_pred_w[-30:]['ds'], 'Pred': prophet_pred_w[-30:]['yhat']})


# In[174]:


prophet_pred_w = prophet_pred_w.set_index('Date')


# In[175]:


prophet_pred_w.index.freq = 'MS'


# In[176]:


prophet_pred_w


# In[177]:


test_w_pr['Prophet_prediction'] = prophet_pred_w['Pred'].values
test_w_pr


# In[178]:


import seaborn as sns


# In[179]:


plt.figure(figsize =(16,5))
ax = sns.lineplot(x = test_w_pr['ds'], y = test_w_pr['Prophet_prediction'])
sns.lineplot(x = test_w_pr['ds'], y = test_w_pr['y']);


# In[191]:


import pickle
Wipro='Wipro.pkl'
f = open(Wipro,'wb')
pickle.dump(model_fit,f)
f.close()


# In[192]:


with open('Wipro.pkl', 'rb') as f:
    data = pickle.load(f)


# In[193]:


data


# In[194]:


get_ipython().system('pip install streamlit')


# In[ ]:




