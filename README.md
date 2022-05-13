# `dmlUtils`: Utilities for `DoubleML`

Utility functions to quickly fit DML models with cross-fitting and
make tables with estimates and RMSEs.

```r
rm(list = ls())
set.seed(42)
libreq(hdm, ggplot2, data.table, knitr,
    DoubleML, mlr3, mlr3learners, mlr3extralearners
  )
theme_set(lal_plot_theme())
source("R/dmlUtils.R")
```

## Data Prep

```r
# %% data prep
data(lalonde.exp)
formula_flex = "re78 ~ treat +
                (poly(age, 4, raw=TRUE) + poly(education, 4, raw=TRUE) +
                poly(re74, 4, raw=TRUE) + poly(re75, 4, raw=TRUE) +
                black + hispanic + married + nodegree + u74 + u75)^2"
model_flex = as.data.table(model.frame(formula_flex, lalonde.exp))
# %%
x_cols = colnames(model_flex)[-c(1,2)]
data_ml = DoubleMLData$new(model_flex, y_col = "re78", d_cols = "treat",
          x_cols = x_cols)

# Classes 'DoubleMLData', 'R6' <DoubleMLData>
#   Public:
#     all_variables: active binding
#     clone: function (deep = FALSE)
#     d_cols: active binding
#     data: active binding
#     data_model: active binding
#     initialize: function (data = NULL, x_cols = NULL, y_col = NULL, d_cols = NULL,
#     n_instr: active binding
#     n_obs: active binding
#     n_treat: active binding
#     other_treat_cols: active binding
#     print: function ()
#     set_data_model: function (treatment_var)
#     treat_col: active binding
#     use_other_treat_as_covariate: active binding
#     x_cols: active binding
#     y_col: active binding
#     z_cols: active binding
#   Private:
#     check_disjoint_sets: function ()
#     d_cols_: treat
#     data_: data.table, data.frame
#     data_model_: data.table, data.frame
#     other_treat_cols_: NULL
#     treat_col_: treat
#     use_other_treat_as_covariate_: TRUE
#     x_cols_: poly.age..4..raw...TRUE..1 poly.age..4..raw...TRUE..2 po ...
#     y_col_: re78
#     z_cols_: NULL
#
```

## Initialise learner objects
```r
# %% learners
lgr::get_logger("mlr3")$set_threshold("warn")
lasso       = lrn("regr.cv_glmnet",    nfolds = 5, s = "lambda.min"); set_threads(lasso)
lasso_class = lrn("classif.cv_glmnet", nfolds = 5, s = "lambda.min"); set_threads(lasso_class)
rf          = lrn("regr.ranger");      set_threads(rf)
rf_class    = lrn("classif.ranger");   set_threads(rf_class)
trees       = lrn("regr.rpart");       set_threads(trees)
trees_class = lrn("classif.rpart");    set_threads(trees_class)
boost       = lrn("regr.glmboost");    set_threads(boost)
boost_class = lrn("classif.glmboost"); set_threads(boost_class)

# <LearnerRegrCVGlmnet:regr.cv_glmnet>
# * Model: -
# * Parameters: family=gaussian, nfolds=5, s=lambda.min
# * Packages: mlr3, mlr3learners, glmnet
# * Predict Type: response
# * Feature types: logical, integer, numeric
# * Properties: selected_features, weights

# <LearnerClassifCVGlmnet:classif.cv_glmnet>
# * Model: -
# * Parameters: nfolds=5, s=lambda.min
# * Packages: mlr3, mlr3learners, glmnet
# * Predict Type: response
# * Feature types: logical, integer, numeric
# * Properties: multiclass, selected_features, twoclass, weights

# <LearnerRegrRanger:regr.ranger>
# * Model: -
# * Parameters: num.threads=6
# * Packages: mlr3, mlr3learners, ranger
# * Predict Type: response
# * Feature types: logical, integer, numeric, character, factor, ordered
# * Properties: hotstart_backward, importance, oob_error, weights

# <LearnerClassifRanger:classif.ranger>
# * Model: -
# * Parameters: num.threads=6
# * Packages: mlr3, mlr3learners, ranger
# * Predict Type: response
# * Feature types: logical, integer, numeric, character, factor, ordered
# * Properties: hotstart_backward, importance, multiclass, oob_error,
#   twoclass, weights

# <LearnerRegrRpart:regr.rpart>: Regression Tree
# * Model: -
# * Parameters: xval=0
# * Packages: mlr3, rpart
# * Predict Type: response
# * Feature types: logical, integer, numeric, factor, ordered
# * Properties: importance, missings, selected_features, weights

# <LearnerClassifRpart:classif.rpart>: Classification Tree
# * Model: -
# * Parameters: xval=0
# * Packages: mlr3, rpart
# * Predict Type: response
# * Feature types: logical, integer, numeric, factor, ordered
# * Properties: importance, missings, multiclass, selected_features,
#   twoclass, weights

# <LearnerRegrGLMBoost:regr.glmboost>
# * Model: -
# * Parameters: list()
# * Packages: mlr3, mlr3extralearners, mboost
# * Predict Type: response
# * Feature types: integer, numeric, factor, ordered
# * Properties: weights

# <LearnerClassifGLMBoost:classif.glmboost>
# * Model: -
# * Parameters: list()
# * Packages: mlr3, mlr3extralearners, mboost
# * Predict Type: response
# * Feature types: integer, numeric, factor, ordered
# * Properties: twoclass, weights
```

## Fit models

```r
# partially linear
lassoPLR = plrFit(data_ml, lasso, lasso_class)
rforsPLR = plrFit(data_ml, rf,    rf_class)
treesPLR = plrFit(data_ml, trees, trees_class)
boostPLR = plrFit(data_ml, boost, boost_class)

# fully nonparametric
lassoIRM = irmFit(data_ml, lasso, lasso_class)
rforsIRM = irmFit(data_ml, rf,    rf_class)
treesIRM = irmFit(data_ml, trees, trees_class)
boostIRM = irmFit(data_ml, boost, boost_class)
```

```r
dmlTab(lassoPLR, rforsPLR, treesPLR, boostPLR, lassoIRM, rforsIRM, treesIRM, boostIRM)
```

||          LASSO    | RF       | CART     |BOOST     | LASSO    | RF       | CART     |BOOST     |
|:--------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
|Estimate | 1889.1452| 1869.2201| 1805.9770| 1709.8805| 1587.8331| 1315.4074| 2140.5224| 1790.9077|
|SE       |  684.0436|  634.5607|  649.6598|  667.7938|  668.2180|  806.9806| 2220.9821|  713.1665|
|RMSE: Y  | 6772.3834| 7180.0214| 6751.6879| 6529.5300| 6573.6237| 6902.0071| 6922.6472| 6777.6399|
|RMSE: D  |    0.4910|    0.5202|    0.5173|    0.4980|    0.4947|    0.5144|    0.5012|    0.4943|
|CE: D    |    0.4135|    0.4472|    0.4247|    0.3978|    0.4292|    0.4562|    0.4022|    0.4270|
