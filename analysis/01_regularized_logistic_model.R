# load libraries ----
library(tidymodels)

# Helper packages
library(readr)       # for importing data
library(themis)      # to balance the dataset
library(vip)         # for variable importance plots
library(tune)
library(glmnet)
library(here)

# set random seed
set.seed(1)

#-------------------------------------------------------------
# survey data
#-------------------------------------------------------------
# main data frame containing, convert characters to factors
# and drop NA values, if any
df_main <- readr::read_csv(here::here("data/survey_drivers.csv")) |>
  mutate(across(where(is.character), as.factor)) |>
  mutate(bbtd = as.factor(bbtd)) |>
  na.omit()

# data splitting: training / testing
splits <- initial_split(df_main, strata = bbtd)
df_train <- training(splits)
df_test  <- testing(splits)

# The data is zero inflated
# this needs adjusting as by, due to the high
# proportion of 0 values by chance the model
# can be considered a "good fit".
# We'll have to down or upsample (impute) the data
# to better balance the data.
df_train |>
  count(bbtd) |>
  mutate(prop = n/sum(n)) |>
  print()

val_set <- initial_validation_split(
  df_train,
  strata = bbtd,
  prop = c(0.6,0.20)
  )

#----------------------------------------------------------------------------
# prediction data
#----------------------------------------------------------------------------
drivers <- terra::rast(here::here("data/drivers_full.tif"))

rwa <- geodata::gadm(
  country="Rwanda",
  level=2,
  path = "data-raw/"
)

# find bounding box of rwanda polygon
rwanda_bbox <- sf::st_bbox(rwa)

# crop to new boundary
drivers_rwa <- terra::crop(
  drivers,
  rwanda_bbox
)

#----------------------------------------------------------------------------
# regularized model
#----------------------------------------------------------------------------
# specify the model ----
# mixture = 1 specifies a pure lasso model
model_specs <- parsnip::logistic_reg(
  penalty = tune(),
  mixture = 1) |>
  parsnip::set_engine("glmnet")

data_specs <- recipes::recipe(bbtd ~ ., data = df_train) |>
  recipes::step_dummy(all_nominal_predictors()) |>
  recipes::step_zv(all_predictors()) |>
  recipes::step_normalize(all_predictors())

workflow_specs <- workflows::workflow() |>
  workflows::add_model(model_specs) |>
  workflows::add_recipe(data_specs)

tgrid <- tibble(penalty = 10^seq(-4, 0, length.out = 30))
folds <- rsample::vfold_cv(df_train, v = 5)

# tune model
lr_res <- workflow_specs |>
  tune::tune_grid(
    df_train,
    resamples = folds,
    grid = tgrid,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc)
  )

top_models <-
  lr_res |>
  show_best(metric = "roc_auc", n = 20) |>
  arrange(desc(mean))

# plot results
lr_plot <- lr_res |>
  tune::collect_metrics() |>
  ggplot(aes(x = penalty, y = mean)) +
  geom_point() +
  geom_line() +
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot <- lr_plot +
  geom_vline(
    xintercept = top_models$penalty[1],
    lty = 2
  )
lr_plot

# several models have equally high performance
# higher penalty is selected
lr_plot <- lr_plot +
  geom_vline(
    xintercept = top_models$penalty[7],
    color = "red"
  )
lr_plot

# model 7 selected (red intercept,
# higher penalty, same performance)
lr_best <-
  lr_res %>%
  collect_metrics() %>%
  arrange(desc(mean)) %>%
  slice(7)

lr_auc <- lr_res |>
  collect_predictions(parameters = lr_best) |>
  roc_curve(bbtd, .pred_0) |>
  mutate(model = "Logistic Regression")

autoplot(lr_auc)

# set the hyperparameter based on the
# grid search, and fit to all training data
best_fit <- tune::finalize_workflow(
  workflow_specs,
  lr_best
) |>
  fit(dat = df_train)

# fit the test data using this model
test_aug <- augment(best_fit, df_test)

class_plot <- ggplot(test_aug,
       aes(bbtd,
           .pred_1)) +
  geom_boxplot() +
  labs (title = "model classification") +
  xlab('BBTD validation dataset') +
  ylab('probability BBTD model prediction') +
  theme_minimal()
class_plot

val_test_summ <- test_aug |>
  group_by(bbtd) |>
  dplyr::summarise(mean_1 = mean(.pred_1),
                   sd_1 = sd(.pred_1))

# get penalty used in best fit
best_fit$fit$fit$spec
best_fit_penalty <- 0.00239502661998749

# get regression coefficients
# zeros are kicked out
coef(extract_fit_engine(best_fit), s = best_fit_penalty)

#----------------------------------------------------------------------------
# prediction
#----------------------------------------------------------------------------
# return probabilities, where each class is
# associated with a layer in an image stack
# and the probabilities reflect the probabilities
# of the classification for said layer
bbtd_probabilities <- terra::predict(
  drivers_rwa,
  best_fit,
  type = "prob"
)

# subset probabilities of disease presence
bbtd_probabilities_1 <- terra::subset(bbtd_probabilities, 2)


# BBTD probability map rwanda
bbtd_probabilities_1_rwanda <-  terra::mask(bbtd_probabilities_1, rwa)

p <- ggplot() +
  tidyterra::geom_spatraster(data = bbtd_probabilities_1_rwanda) +
  scale_fill_distiller(
    palette = 'YlOrBr',
    direction = 1,
    na.value = NA
    ) +
  geom_sf(
    data = rwa,
    color = 'grey80',
    fill = NA,
    lwd = 0.3
  ) +
  geom_sf(
    data = rwa |> sf::st_as_sf() |> sf::st_union(),
    color = 'grey80',
    fill = NA,
    lwd = 1
  ) +
  labs (title = "Probability of BBTD presence",
        subtitle = "Penalized Logistic Regression Model",
        fill = "class probabilities") +
  scale_x_continuous(limits = c(28.8, 30.9)) +
  scale_y_continuous(limits = c(-3, -0.9)) +
  theme_minimal() +
  theme(
    legend.position = "bottom"
  )

print(p)

