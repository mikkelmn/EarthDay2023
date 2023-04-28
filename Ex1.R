library(ncdf4)
library(tidyverse)
library(forecast)
library(keras)
library(tensorflow)

# If you are running an ARM mac you should not install Python within R from
# the installer provided by the Keras package. Instead your Python directory 
# should be used in the below. Here, Mikkel's is used since he uses an M1 mac
use_python("/Users/mikkelmandrup/miniconda3/bin/python3")

# For loading the data
data = nc_open(filename = "temp_anomaly_ex1.nc")

lat <- ncvar_get(data, "lat")
nlat <- dim(lat) # 10

lon <- ncvar_get(data, "lon")
nlon <- dim(lon) # 10

time <- ncvar_get(data, "time") #months from January 1880
nt <- dim(time) # 1420


# The array of tempanomalies
tempanomaly_array <- ncvar_get(data, "tempanomaly") 

# Making the missing values NAs
tempanomaly_array[tempanomaly_array == 32767] <- NA

tempanomaly_array[ , , 1410] # view and example

temp_series = tempanomaly_array[6,6,]
plot(temp_series, type = "l")
ggAcf(temp_series)

# CCF very correlated
ccf(tempanomaly_array[6,6,-(1408:1420)], tempanomaly_array[5,5,-(1408:1420)])

# response variable
response = tempanomaly_array[ 6, 6, 1:1407]


input = tempanomaly_array[ , , 1:1420] %>% as.vector()

input = map(.x = seq(100, 142000, 100), .f = function(x) input[(x-99):x])

input = map(.x = 1:1420, .f = function(x) input[[x]][-56]) %>% flatten_dbl() %>% matrix(ncol = 1420)

pred_input = input[, 1408:1420]
input = input[, 1:1407]

set.seed(1234) # to reproduce the splits
index <- sample(2, 
                size = length(response),  
                replace = TRUE, 
                prob = c(0.80, 0.20))

x_train = input[, index == 1]
x_val = input[, index == 2]

y_train = response[index == 1]
y_val = response[index == 2]

x_val = (x_val - min(x_train)) / (max(x_train) - min(x_train))
x_pred = (pred_input - min(x_train)) / (max(x_train) - min(x_train))
x_train = (x_train - min(x_train)) / (max(x_train) - min(x_train))

model <- keras_model_sequential()
model %>% layer_dense(units = 50, activation = "relu", input_shape = c(99)) %>% 
  layer_dense(units = 25, activation = "relu") %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = 1)



opt = optimizer_adam(learning_rate = 0.001)
model %>% compile(loss = 'mse',
                  optimizer = opt,
                  metrics = 'mae')

history <- model %>% fit(
  x = t(x_train),
  y = y_train,
  batch_size = 32,
  epochs = 500, 
  validation_data = list(t(x_val), y_val)
)

pred = predict(model, t(x_pred))
write.csv(x = pred, file = "pred_Ex1_MMN")





