# mainly a port of
# https://github.com/pangeo-data/WeatherBench/blob/master/notebooks/3-cnn-example.ipynb

library(reticulate)
library(tensorflow)
library(keras)
library(tfdatasets)
library(tfautograph)

builtins <- import_builtins(convert = FALSE)

xr <- import("xarray")
z500 <-
  xr$open_mfdataset("../weatherbench/geopotential_500/*.nc", combine = "by_coords")
z500
z500$z

z500$z$isel(time = 0L)$plot()

climatology <- z500$sel(time = "2016")$mean('time')$load()
climatology$z$plot()


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

lead_time <- 6
n_samples <- dim(train)[1] - lead_time
batch_size <- 32

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
  dataset_shuffle(buffer_size = n_samples) %>%
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

valid_loss <- tf$keras$metrics$Mean(name='test_loss')

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

training_loop <- tf_function(autograph(function(train_ds, valid_ds) {
  
  for (train_batch in train_ds) {
    train_step(train_batch)
  }
  
  for (valid_batch in valid_ds) {
    valid_step(valid_batch[[1]], valid_batch[[2]])
  }
  
  tf$print("MSE: train: ", train_loss$result(), ", validation: ", valid_loss$result())
  
  train_loss$reset_states()
  valid_loss$reset_states()
  
}))

for (epoch in 1:5) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(train_ds, valid_ds)  
}




