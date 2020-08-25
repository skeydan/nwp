# mainly a port of
# https://github.com/pangeo-data/WeatherBench/blob/master/notebooks/3-cnn-example.ipynb

library(reticulate)
library(tensorflow)
library(keras)
library(tfdatasets)
library(tfautograph)
tb <- import("tensorboard")
library(tidyverse)

builtins <- import_builtins(convert = FALSE)
xr <- import("xarray")

# Data prep ---------------------------------------------------------------


load_data <- function(path, var, years) {
  ds <-
    xr$open_mfdataset(paste0(path, "*.nc"), combine = 'by_coords')[[var]]
  ds <- ds$drop('level')
  ds$sel(time = years)
}

path = '../weatherbench/geopotential_500/'
z500_train <-
  load_data(path, "z", years =  builtins$slice("2015", '2015'))
z500_valid <-
  load_data(path, "z", years =  builtins$slice("2016", '2016'))
z500_test <-
  load_data(path, "z", years =  builtins$slice("2017", '2018'))



path = '../weatherbench/temperature_850/'
t850_train <-
  load_data(path, "t", years =  builtins$slice("2015", '2015'))
t850_valid <-
  load_data(path, "t", years =  builtins$slice("2016", '2016'))
t850_test <-
  load_data(path, "t", years =  builtins$slice("2017", '2018'))


train <-
  abind::abind(z500_train$values, t850_train$values, along = 4)
level_means <- apply(train, 4, mean)
level_sds <- apply(train, 4, sd)
train[, , , 1] <- (train[, , , 1] - level_means[1]) / level_sds[1]
train[, , , 2] <- (train[, , , 2] - level_means[2]) / level_sds[2]

valid <-
  abind::abind(z500_valid$values, t850_valid$values, along = 4)
valid[, , , 1] <- (valid[, , , 1] - level_means[1]) / level_sds[1]
valid[, , , 2] <- (valid[, , , 2] - level_means[2]) / level_sds[2]

test <- abind::abind(z500_test$values, t850_test$values, along = 4)
test[, , , 1] <- (test[, , , 1] - level_means[1]) / level_sds[1]
test[, , , 2] <- (test[, , , 2] - level_means[2]) / level_sds[2]

lead_time <- 3 * 24 # 3d
batch_size <- 32

n_samples <- dim(train)[1] - lead_time
train_x <- train %>%
  tensor_slices_dataset() %>%
  dataset_take(n_samples)

train_y <- train %>%
  tensor_slices_dataset() %>%
  dataset_skip(lead_time)

train_ds <- zip_datasets(train_x, train_y) %>%
  dataset_shuffle(buffer_size = n_samples) %>%
  dataset_batch(batch_size = batch_size, drop_remainder = TRUE)

b <- as_iterator(train_ds) %>% iter_next()
b[[1]][ , 1, 1, 2]
b[[2]][ , 1, 1, 2]

n_samples <- dim(valid)[1] - lead_time
valid_x <- valid %>%
  tensor_slices_dataset() %>%
  dataset_take(n_samples)

valid_y <- valid %>%
  tensor_slices_dataset() %>%
  dataset_skip(lead_time)

valid_ds <- zip_datasets(valid_x, valid_y) %>%
  dataset_batch(batch_size = batch_size, drop_remainder = TRUE)

n_samples <- dim(test)[1] - lead_time
test_x <- test %>%
  tensor_slices_dataset() %>%
  dataset_take(n_samples)

test_y <- test %>%
  tensor_slices_dataset() %>%
  dataset_skip(lead_time)

test_ds <- zip_datasets(test_x, test_y) %>%
  dataset_batch(batch_size = batch_size, drop_remainder = TRUE)


# Model -------------------------------------------------------------------


periodic_padding_2d <- function(pad_width,
                                name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$pad_width <- pad_width
    
    function (x, mask = NULL) {
      x <- if (self$pad_width == 0) {
        x
      } else {
        lon_dim <- dim(x)[3]
        pad_width <- tf$cast(self$pad_width, tf$int32)
        # wrap around for longitude
        tf$concat(list(x[, ,-pad_width:lon_dim,],
                       x,
                       x[, , 1:pad_width,]),
                  axis = 2L) %>%
          tf$pad(list(
            list(0L, 0L),
            # zero-pad for latitude
            list(pad_width, pad_width),
            list(0L, 0L),
            list(0L, 0L)
          ))
      }
    }
  })
}

periodic_conv_2d <- function(filters,
                             kernel_size,
                             name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$padding <- periodic_padding_2d(pad_width = (kernel_size - 1) / 2)
    self$conv <-
      layer_conv_2d(filters = filters,
                    kernel_size = kernel_size,
                    padding = 'valid')
    
    function (x, mask = NULL) {
      x %>% self$padding() %>% self$conv()
    }
  })
}


test_pp <- periodic_padding_2d(pad_width = 2)
test_t <- tf$random$normal(c(8L, 32L, 64L, 1L))
# shape=(8, 36, 68, 1)
test_pp(test_t)
test_pc <- periodic_conv_2d(filters = 16, kernel_size = 5)
# shape=(8, 32, 64, 16)
test_pc(test_t)

periodic_cnn <- function(filters = c(64, 64, 64, 64, 2),
                         kernel_size = c(5, 5, 5, 5, 5),
                         dropout = rep(0, 5),
                         name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$conv1 <-
      periodic_conv_2d(filters = filters[1], kernel_size = kernel_size[1])
    self$act1 <- layer_activation_leaky_relu()
    self$drop1 <- layer_dropout(rate = dropout[1])
    self$conv2 <-
      periodic_conv_2d(filters = filters[2], kernel_size = kernel_size[2])
    self$act2 <- layer_activation_leaky_relu()
    self$drop2 <- layer_dropout(rate =dropout[2])
    self$conv3 <-
      periodic_conv_2d(filters = filters[3], kernel_size = kernel_size[3])
    self$act3 <- layer_activation_leaky_relu()
    self$drop3 <- layer_dropout(rate = dropout[3])
    self$conv4 <-
      periodic_conv_2d(filters = filters[4], kernel_size = kernel_size[4])
    self$act4 <- layer_activation_leaky_relu()
    self$drop4 <- layer_dropout(rate = dropout[4])
    self$conv5 <-
      periodic_conv_2d(filters = filters[5], kernel_size = kernel_size[5])
    
    function (x, mask = NULL) {
      x %>%
        self$conv1() %>%
        self$act1() %>%
        self$drop1() %>%
        self$conv2() %>%
        self$act2() %>%
        self$drop2() %>%
        self$conv3() %>%
        self$act3() %>%
        self$drop3() %>%
        self$conv4() %>%
        self$act4() %>%
        self$drop4() %>%
        self$conv5()
    }
  })
}

model <- periodic_cnn()

# Training ----------------------------------------------------------------

loss <- tf$keras$losses$MeanSquaredError(reduction = tf$keras$losses$Reduction$SUM)
optimizer <- optimizer_adam()

train_loss <- tf$keras$metrics$Mean(name='train_loss')

valid_loss <- tf$keras$metrics$Mean(name='valid_loss')

train_step <- function(train_batch) {

  with (tf$GradientTape() %as% tape, {
    predictions <- model(train_batch[[1]])
    l <- loss(train_batch[[2]], predictions)
  })

  gradients <- tape$gradient(l, model$trainable_variables)
  optimizer$apply_gradients(purrr::transpose(list(
    gradients, model$trainable_variables
  )))

  train_loss(l)
  
}

valid_step <- function(valid_batch) {
  predictions <- model(valid_batch[[1]])
  l <- loss(valid_batch[[2]], predictions)
  
  valid_loss(l)
}

training_loop <- tf_function(autograph(function(train_ds, valid_ds, epoch) {
  
  for (train_batch in train_ds) {
    train_step(train_batch)
  }
  
  for (valid_batch in valid_ds) {
    valid_step(valid_batch)
  }
  
  #tf$print("MSE: train: ", train_loss$result(), ", validation: ", valid_loss$result())
  with (writer$as_default(), {
    tf$summary$scalar("train_loss", train_loss$result(), epoch)
    tf$summary$scalar("valid_loss", valid_loss$result(), epoch)
  })
  
  train_loss$reset_states()
  valid_loss$reset_states()
  
}))

unlink("logs", recursive = TRUE)
writer <- tf$summary$create_file_writer("logs")

n_epochs <- 1

for (epoch in 1:n_epochs) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(train_ds, valid_ds, epoch)  
  writer$flush()
}

acc <- tb$backend$event_processing$event_accumulator$EventAccumulator("logs")
acc$Reload()
acc$Tags()
train_losses <- purrr::map(acc$Tensors("train_loss"), function(t) tf$make_ndarray(t$tensor_proto))
valid_losses <- purrr::map(acc$Tensors("valid_loss"), function(t) tf$make_ndarray(t$tensor_proto))
train_losses
valid_losses

history <- data.frame(epoch = as.factor(1:n_epochs), training = unlist(train_losses), validation = unlist(valid_losses)) 
history %>% pivot_longer(-epoch, names_to = "phase") %>%
  ggplot(aes(x = epoch, y = value, color = phase)) + geom_point() +
  theme_classic() +
  scale_color_manual(values = c("#00FF7F", "#593780")) +
  ggtitle("Mean squared error (training/validation, lead time = 3 days)")


# Metrics -----------------------------------------------------------------

deg2rad <- function(d) {
  (d / 180) * pi
}

# Latitude weighted root mean squared error
weighted_rmse <- function(forecast, ground_truth) {
  error <- forecast$values - ground_truth$values
  weights_lat <- cos(deg2rad(ground_truth$lat$values))
  weights_lat <- weights_lat / mean(weights_lat)
  np <- import("numpy", convert = FALSE)
  wl <- r_to_py(weights_lat)$reshape(c(1L,1L,32L,1L))
  e <- r_to_py(error)
  np$sqrt((np$multiply(np$square(e), wl))$mean(axis = tuple(1L, 2L, 3L)))
}


# Weekly climatology ----------------------------------------------------

train_xr <- xr$merge(list(t850_train, z500_train))
# https://en.wikipedia.org/wiki/ISO_week_date

weekly_averages <- train_xr$groupby("time.week")$mean("time")

test_xr <- xr$merge(list(t850_test, z500_test))
test_time <- test_xr$time

fc_list <- vector(mode = "list", length = test_time$size)
for (t in 1:test_time$size) {
  fc_list[[t]] <- weekly_averages$sel(week = test_time[t-1]$time.week)
}

weekly_clim_preds <- xr$concat(fc_list, dim = test_time)

wrmse <- weighted_rmse(weekly_clim_preds$to_array(), test_xr$to_array())
# result reported on github uses weekly climatology from whole dataset
#[  4.0963539  983.51442889]



# persistence forecast ----------------------------------------------------

persistence_forecast <- test_xr$isel(time = builtins$slice(0L, as.integer(-lead_time)))
sel <- test_xr$isel(time = builtins$slice(as.integer(lead_time), test_xr$time$size))
wrmse <- weighted_rmse(persistence_forecast$to_array(), sel$to_array())
# [  4.2913616  935.91602924]


# CNN forecasts -------------------------------------------------------------

test_loss <- tf$keras$metrics$Mean(name='test_loss')

preds_list <- vector(mode = "list", length = dim(test)[1])

test_step <- function(test_batch) {
  predictions <- model(test_batch[[1]])
  l <- loss(test_batch[[2]], predictions)
  
  test_loss(l)
}

test_iterator <- as_iterator(test_ds) 
while (TRUE) {
  
}

# Unnormalize


