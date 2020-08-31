# mainly a port of
# https://github.com/pangeo-data/WeatherBench/blob/master/notebooks/3-cnn-example.ipynb

Sys.setenv(CUDA_VISIBLE_DEVICES=-1)

library(reticulate)
library(tensorflow)
library(keras)
library(tfdatasets)
library(tfautograph)
tb <- import("tensorboard")

library(tidyverse)
library(lubridate)
# https://ropensci.org/blog/2019/11/05/tidync/
library(tidync)

# Data prep ---------------------------------------------------------------

path <- '../weatherbench/'

tidync(paste0(
  path,
  "geopotential_500/geopotential_500hPa_2015_5.625deg.nc"
)) %>% hyper_array()
z500_train <-
  (tidync(
    paste0(
      path,
      "geopotential_500/geopotential_500hPa_2015_5.625deg.nc"
    )
  ) %>%
    hyper_array())[[1]]
dim(z500_train)
image(
  z500_train[, , 1],
  col = hcl.colors(20, "viridis"),
  xaxt = 'n',
  yaxt = 'n',
  main = "500hPa geopotential"
)

tidync(paste0(path, "temperature_850/temperature_850hPa_2015_5.625deg.nc")) %>% hyper_array()
t850_train <-
  (tidync(
    paste0(path, "temperature_850/temperature_850hPa_2015_5.625deg.nc")
  ) %>%
    hyper_array())[[1]]
image(
  t850_train[, , 1],
  col = hcl.colors(20, "YlOrRd", rev = TRUE),
  xaxt = 'n',
  yaxt = 'n',
  main = "850hPa temperature"
)

z500_valid <-
  (tidync(
    paste0(
      path,
      "geopotential_500/geopotential_500hPa_2016_5.625deg.nc"
    )
  ) %>%
    hyper_array())[[1]]
t850_valid <-
  (tidync(
    paste0(path, "temperature_850/temperature_850hPa_2016_5.625deg.nc")
  ) %>%
    hyper_array())[[1]]

z500_test <-
  (tidync(
    paste0(
      path,
      "geopotential_500/geopotential_500hPa_2017_5.625deg.nc"
    )
  ) %>%
    hyper_array())[[1]]
t850_test <-
  (tidync(
    paste0(path, "temperature_850/temperature_850hPa_2017_5.625deg.nc")
  ) %>%
    hyper_array())[[1]]

train_all <- abind::abind(z500_train, t850_train, along = 4)
dim(train_all)
train_all <- aperm(train_all, perm = c(3, 2, 1, 4))
dim(train_all)

level_means <- apply(train_all, 4, mean)
level_means
level_sds <- apply(train_all, 4, sd)
level_sds

train <- train_all
train[, , , 1] <- (train[, , , 1] - level_means[1]) / level_sds[1]
train[, , , 2] <- (train[, , , 2] - level_means[2]) / level_sds[2]


valid_all <- abind::abind(z500_valid, t850_valid, along = 4)
valid_all <- aperm(valid_all, perm = c(3, 2, 1, 4))

valid <- valid_all
valid[, , , 1] <- (valid[, , , 1] - level_means[1]) / level_sds[1]
valid[, , , 2] <- (valid[, , , 2] - level_means[2]) / level_sds[2]

test_all <- abind::abind(z500_test, t850_test, along = 4)
test_all <- aperm(test_all, perm = c(3, 2, 1, 4))

test <- test_all
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
b[[1]][, 1, 1, 1]
b[[2]][, 1, 1, 1]

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
        tf$concat(list(x[, , -pad_width:lon_dim, ],
                       x,
                       x[, , 1:pad_width, ]),
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
    self$padding <-
      periodic_padding_2d(pad_width = (kernel_size - 1) / 2)
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
                         dropout = rep(0.2, 5),
                         name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$conv1 <-
      periodic_conv_2d(filters = filters[1], kernel_size = kernel_size[1])
    self$act1 <- layer_activation_leaky_relu()
    self$drop1 <- layer_dropout(rate = dropout[1])
    self$conv2 <-
      periodic_conv_2d(filters = filters[2], kernel_size = kernel_size[2])
    self$act2 <- layer_activation_leaky_relu()
    self$drop2 <- layer_dropout(rate = dropout[2])
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

loss <-
  tf$keras$losses$MeanSquaredError(reduction = tf$keras$losses$Reduction$SUM)
optimizer <- optimizer_adam()

train_loss <- tf$keras$metrics$Mean(name = 'train_loss')

valid_loss <- tf$keras$metrics$Mean(name = 'valid_loss')

train_step <- function(train_batch) {
  with (tf$GradientTape() %as% tape, {
    predictions <- model(train_batch[[1]])
    l <- loss(train_batch[[2]], predictions)
  })
  
  gradients <- tape$gradient(l, model$trainable_variables)
  optimizer$apply_gradients(purrr::transpose(list(gradients, model$trainable_variables)))
  
  train_loss(l)
  
}

valid_step <- function(valid_batch) {
  predictions <- model(valid_batch[[1]])
  l <- loss(valid_batch[[2]], predictions)
  
  valid_loss(l)
}

training_loop <-
  tf_function(autograph(function(train_ds, valid_ds) {
    for (train_batch in train_ds) {
      train_step(train_batch)
    }
    
    for (valid_batch in valid_ds) {
      valid_step(valid_batch)
    }
    
    tf$print("MSE: train: ", train_loss$result(), ", validation: ", valid_loss$result())

    train_loss$reset_states()
    valid_loss$reset_states()
    
  }))

n_epochs <- 10

for (epoch in 1:n_epochs) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(train_ds, valid_ds)
}

# Metrics -----------------------------------------------------------------

deg2rad <- function(d) {
  (d / 180) * pi
}

# Latitude weighted root mean squared error
lats <-
  tidync(paste0(
    path,
    "geopotential_500/geopotential_500hPa_2015_5.625deg.nc"
  ))$transforms$lat %>%
  select(lat) %>%
  pull()
lats
lat_weights <- cos(deg2rad(lats))
lat_weights <- lat_weights / mean(lat_weights)

weighted_rmse <- function(forecast, ground_truth) {
  error <- (forecast - ground_truth) ^ 2
  for (i in seq_along(lat_weights)) {
    error[, i, ,] <- error[, i, ,] * lat_weights[i]
  }
  apply(error, 4, mean) %>% sqrt()
}


# Weekly climatology ----------------------------------------------------
train_file <-
  paste0(path,
         "geopotential_500/geopotential_500hPa_2015_5.625deg.nc")

times_train <-
  (tidync(train_file) %>% activate("D2") %>% hyper_array())$time

time_unit_train <- ncmeta::nc_atts(train_file, "time") %>%
  tidyr::unnest(cols = c(value)) %>%
  dplyr::filter(name == "units")

time_unit_train
time_parts_train <-
  RNetCDF::utcal.nc(time_unit_train$value, times_train)

iso_train <- ISOdate(
  time_parts_train[, "year"],
  time_parts_train[, "month"],
  time_parts_train[, "day"],
  time_parts_train[, "hour"],
  time_parts_train[, "minute"],
  time_parts_train[, "second"]
)

isoweeks_train <- map(iso_train, isoweek) %>% unlist()

train_by_week <- apply(train_all, c(2, 3, 4), function(x) {
  tapply(x, isoweeks_train, function(y) {
    mean(y)
  })
})
dim(train_by_week)

test_file <-
  paste0(path,
         "geopotential_500/geopotential_500hPa_2017_5.625deg.nc")

times_test <-
  (tidync(test_file) %>% activate("D2") %>% hyper_array())$time

time_unit_test <- ncmeta::nc_atts(test_file, "time") %>%
  tidyr::unnest(cols = c(value)) %>%
  dplyr::filter(name == "units")

time_unit_test
time_parts_test <-
  RNetCDF::utcal.nc(time_unit_test$value, times_test)

iso_test <- ISOdate(
  time_parts_test[, "year"],
  time_parts_test[, "month"],
  time_parts_test[, "day"],
  time_parts_test[, "hour"],
  time_parts_test[, "minute"],
  time_parts_test[, "second"]
)

isoweeks_test <- map(iso_test, isoweek) %>% unlist()

climatology_forecast <- test_all

for (i in 1:dim(climatology_forecast)[1]) {
  week <- isoweeks_test[i]
  lookup <- train_by_week[week, , , ]
  climatology_forecast[i, , ,] <- lookup
}

wrmse <-
  weighted_rmse(climatology_forecast, test_all)
# result reported on github uses weekly climatology from whole dataset
# 974.499176   4.092189


# persistence forecast ----------------------------------------------------

persistence_forecast <-
  test_all[1:(dim(test_all)[1] - lead_time), , ,]
test_period <- test_all[(lead_time + 1):dim(test_all)[1], , ,]
wrmse <- weighted_rmse(persistence_forecast, test_period)
# 937.549349   4.319022

# CNN forecasts -------------------------------------------------------------

test_wrmses <- data.frame()

test_loss <- tf$keras$metrics$Mean(name = 'test_loss')

test_step <- function(test_batch, batch_index) {
  predictions <- model(test_batch[[1]])
  l <- loss(test_batch[[2]], predictions)
  
  predictions <- predictions %>% as.array()
  predictions[, , , 1] <- predictions[, , , 1] * level_sds[1] + level_means[1]
  predictions[, , , 2] <- predictions[, , , 2] * level_sds[2] + level_means[2]
  
  wrmse <- weighted_rmse(predictions, test_all[batch_index:(batch_index + 31), , ,])
  test_wrmses <<- test_wrmses %>% bind_rows(c(z = wrmse[1], temp = wrmse[2]))
  
  test_loss(l)
}

test_iterator <- as_iterator(test_ds)

batch_index <- 0
while (TRUE) {
  test_batch <- test_iterator %>% iter_next()
  if (is.null(test_batch))
    break
  batch_index <- batch_index + 1
  test_step(test_batch, as.integer(batch_index))
}

test_loss$result() %>% as.numeric()

test_wrmses$z  %>% summary()

test_wrmses$temp  %>% summary()

apply(test_wrmses, 2, mean) %>% round(2)
# weatherbench: ~ ca. 1100 / 5 (lead time 72)
