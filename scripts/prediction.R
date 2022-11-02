require(tidyverse)
require(keras)
require(tensorflow)
load('data/claims-test.RData')
source('scripts/preprocessing.R')
tf_model <- load_model_tf('results/model')

# apply preprocessing pipeline
clean_df <- claims_test %>%
  slice(1:100) %>%
  parse_data() %>%
  select(.id, text_clean)

# grab input
x <- clean_df %>%
  pull(text_clean)

# compute predictions
preds <- predict(tf_model, x) %>%
  as.numeric()

# export (KEEP THIS FORMAT IDENTICAL)
pred_df <- clean_df %>%
  bind_cols(pred = preds) %>%
  select(.id, pred)

save(pred_df, file = 'results/preds.RData')
