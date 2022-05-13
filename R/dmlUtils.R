# p_load(DoubleML, mlr3, mlr3learners)
# %%
plrFit = function(dat, outcomeMod, pscoreMod, nf = 3){
  #' Wrapper for Partially Linear Model
  #' @param dat A DoubleMLData object initialised using using `DoubleMLData$new`
  #' @param outcomeMod A mlr3 learner object for the outcome model
  #' @param pscoreMod  A mlr3 learner object for the treatment model
  #' @param nf Number of folds for cross-fitting
  #' @return list with (estimate, SE, outcome RMSE, treatment RMSE, treatment Classification Error)
  #' @references Bach, P., V. Chernozhukov, M. S. Kurz, and M. Spindler. (2021):
  #' “DoubleML -- An Object-Oriented Implementation of Double Machine Learning
  #' in R,” arXiv [stat.ML],.
  #' @export
  ############################################################
  # store data
  y = as.matrix(dat$data[, 1]); d = as.matrix(dat$data[, 2])
  ############################################################
  # init model class
  dml_plr = DoubleMLPLR$new(dat, ml_g = outcomeMod, ml_m = pscoreMod, n_folds=nf)
  # fit it
  dml_plr$fit(store_predictions=TRUE)
  est = dml_plr$coef; se = dml_plr$se
  ############################################################
  # accuracy measures
  ############################################################
  mu_hat = as.matrix(dml_plr$predictions$ml_g) # predictions of g_o
  pi_hat = as.matrix(dml_plr$predictions$ml_m) # predictions of m_o
  # outcome and treatment are in
    # cross-fitted RMSE: outcome
  predictions_y = as.matrix(d*est) + mu_hat # predictions for y
  y_rmse  = sqrt(mean((y - predictions_y)^2))
  # cross-fitted RMSE: treatment
  d_rmse = sqrt(mean((d-pi_hat)^2))
  # cross-fitted classification error : treatment
  d_ce = mean(ifelse(pi_hat > 0.5, 1, 0) != d)
  ############################################################
  list(est = est, se = se, outcome_rmse = y_rmse, treatment_rmse = d_rmse,
      treatment_ce = d_ce)
}

# %%
irmFit = function(dat, outcomeMod, pscoreMod, trim_thres = 0.01, nf = 3){
  #' Wrapper for Nonparametric IRM Model
  #' @param dat A DoubleMLData object initialised using using `DoubleMLData$new`
  #' @param outcomeMod A mlr3 learner object for the outcome model
  #' @param pscoreMod  A mlr3 learner object for the treatment model
  #' @param trim_thres Threshold for trimming extreme propensity scores
  #' @param nf Number of folds for cross-fitting
  #' @return list with (estimate, SE, outcome RMSE, treatment RMSE, treatment Classification Error)
  #' @references Bach, P., V. Chernozhukov, M. S. Kurz, and M. Spindler. (2021):
  #' “DoubleML -- An Object-Oriented Implementation of Double Machine Learning
  #' in R,” arXiv [stat.ML],.
  #' @export
  # outcome and treatment are in data object
  ############################################################
  # store data
  y = as.matrix(dat$data[, 1]); d = as.matrix(dat$data[, 2])
  # init model object
  dml_irm = DoubleMLIRM$new(dat, ml_g = outcomeMod, ml_m = pscoreMod,
                  trimming_threshold = trim_thres, n_folds=nf)
  # fit it
  dml_irm$fit(store_predictions=TRUE)
  est = dml_irm$coef; se = dml_irm$se
  ############################################################
  # accuracy measures
  ############################################################
  mu0_hat = as.matrix(dml_irm$predictions$ml_g0) # predictions of g_0(D=0, X)
  mu1_hat = as.matrix(dml_irm$predictions$ml_g1) # predictions of g_0(D=1, X)
  pi_hat  = as.matrix(dml_irm$predictions$ml_m) # predictions of m_o
  # cross-fitted RMSE: outcome
  predictions_y = d*mu1_hat+(1-d)*mu0_hat # predictions of g_0
  y_rmse  = sqrt(mean((y - predictions_y)^2))
  # cross-fitted RMSE: treatment
  d_rmse = sqrt(mean((d-pi_hat)^2))
  # cross-fitted classification error : treatment
  d_ce = mean(ifelse(pi_hat > 0.5, 1, 0) != d)
  ############################################################
  list(est = est, se = se, outcome_rmse = y_rmse, treatment_rmse = d_rmse,
      treatment_ce = d_ce)
}

# %%
dmlTab = \(..., modsel = F){
  #' Table maker for DML
  #' @param ... list of `plrFit` or `irmFit` output lists of arbitrary length
  #' @param modsel Boolean for whether to return indices of best outcome and treatment models
  #' @export
  mods = list(...)
  tmp = Map(as.numeric, mods)
  table = do.call(cbind, tmp)
  rownames(table) = c("Estimate", "SE","RMSE: Y","RMSE: D", "CE: D")
  if(modsel)
    return(list(table, bestY = which.min(table[3,]),bestD = which.min(table[5,])))
  else
    return(table)
}

# %%
