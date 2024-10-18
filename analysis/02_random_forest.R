# load libraries ----
library(tidymodels)

# Helper packages
library(readr)       # for importing data
library(themis)      # to balance the dataset
library(vip)         # for variable importance plots
library(ranger)

#-------------------------------------------------------------
# survey data
#-------------------------------------------------------------
# main data frame containing, convert characters to factors
# and drop NA values, if any
df_main <- readr::read_csv(here::here("data/survey_drivers.csv")) |>
  mutate(across(where(is.character), as.factor)) |>
  mutate(bbtd = as.factor(bbtd)) |>
  na.omit()

splits <- initial_split(df_main, strata = bbtd)
df_train <- training(splits)
df_test  <- testing(splits)

df_train |>
  count(bbtd) |>
  mutate(prop = n/sum(n)) |>
  print()

val_set <- initial_validation_split(
  df_train,
  strata = bbtd,
  prop = c(0.6,0.20)
)

folds <- rsample::vfold_cv(df_train, v = 10)
#-------------------------------------------------------------
# prediction data
#-------------------------------------------------------------
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

#-------------------------------------------------------------
# Random forest model
#-------------------------------------------------------------
# number of cores on your computer
cores <- parallel::detectCores()
cores

rf_mod <-
  parsnip::rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 1000
  ) %>%
  set_engine("ranger", num.threads = cores) %>%
  set_mode("classification")

# create the recipe and workflow
rf_recipe <-
  recipes::recipe(bbtd ~ ., data = df_main)

rf_workflow <-
  workflows::workflow() %>%
  workflows::add_model(rf_mod) %>%
  workflows::add_recipe(rf_recipe)

rf_mod
extract_parameter_set_dials(rf_mod)

set.seed(345)
rf_res <-
  rf_workflow %>%
  tune::tune_grid(
    folds,
    grid = 25,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc)
    )

# top 5 random forest models,
# out of the 25 candidates
rf_res %>%
  show_best(metric = "roc_auc")

autoplot(rf_res)

# select the best model
# according to the ROC AUC metric
rf_best <-
  rf_res %>%
  select_best(metric = "roc_auc")

rf_best_hp <- tune::finalize_workflow(
  rf_workflow,
  rf_best
)

# train a final (best) model with optimal
# hyper-parameters
rf_best_model <- fit(rf_best_hp, df_train)

#----------------------------------------------------------------------------
# prediction
#----------------------------------------------------------------------------
bbtd_probabilities <- terra::predict(
  drivers_rwa,
  rf_best_model,
  type = "prob",
  na.rm = TRUE
  )

# subset probabilities of disease presence
bbtd_probabilities_1 <- terra::subset(bbtd_probabilities, 2)
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
        subtitle = "random forest",
        fill = "class probabilities") +
  scale_x_continuous(limits = c(28.8, 30.9)) +
  scale_y_continuous(limits = c(-3, -0.9)) +
  theme_minimal() +
  theme(
    legend.position = "bottom"
  )

print(p)
