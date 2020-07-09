mase_cal <- function(insample, outsample, forecasts) {
  stopifnot(stats::is.ts(insample))
  #Used to estimate MASE
  frq <- stats::frequency(insample)
  forecastsNaiveSD <- rep(NA,frq)
  for (j in (frq+1):length(insample)){
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j-frq])
  }
  masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)

  outsample <- as.numeric(outsample) ; forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample-forecasts))/masep
  return(mase)
}

calc_errors_tps <- function(dataset) {

  total_snaive_errors <- c(0,0)
  for (i in 1:length(dataset)) {
    tryCatch({
    lentry <- dataset[[i]]
    insample <- lentry$x

    #extrac forecasts and attach the snaive for completion
    ff <- lentry$ff
    ensemble = apply(ff, 2, mean)
    names(ensemble) = 'ensemble'
        
    ff <- rbind(ff, ensemble, snaive_forec(insample, lentry$h))
    colnames(ff) = NULL
        
    frq <- frq <- stats::frequency(insample)
    insample <- as.numeric(insample)
    outsample <- as.numeric(lentry$xx)
    masep <- mean(abs(utils::head(insample,-frq) - utils::tail(insample,-frq)))

    repoutsample <- matrix(
      rep(outsample, each=nrow(ff)),
      nrow=nrow(ff))

#     smape_err <- 200*abs(ff - repoutsample) / (abs(ff) + abs(repoutsample))

    error <- ff - repoutsample
    lentry$error <- error[-nrow(error),]
        
    mase_err <- abs(ff - repoutsample) / masep

    lentry$snaive_mase <- mase_err[nrow(mase_err), ]
#     lentry$snaive_smape <- smape_err[nrow(smape_err),]

    lentry$mase_err <- mase_err[-nrow(mase_err),]
#     lentry$smape_err <- smape_err[-nrow(smape_err),]
    
    lentry$ff <- ff[-nrow(ff),]
        
    dataset[[i]] <- lentry
#     total_snaive_errors <- total_snaive_errors + c(mean(lentry$snaive_mase),
#                                                    mean(lentry$snaive_smape))
    } , error = function (e) {
      print(paste("Error when processing OWIs in series: ", i))
      print(e)
      e
    })
  }
#   total_snaive_errors = total_snaive_errors / length(dataset)
#   avg_snaive_errors <- list(avg_mase=total_snaive_errors[1],
#                             avg_smape=total_snaive_errors[2])


  for (i in 1:length(dataset)) {
    lentry <- dataset[[i]]
#     dataset[[i]]$errors <- 0.5*(rowMeans(lentry$mase_err)/avg_snaive_errors$avg_mase +
#                                   rowMeans(lentry$smape_err)/avg_snaive_errors$avg_smape)
    dataset[[i]]$errors <- rowMeans(lentry$mase_err)
    dataset[[i]]$sderror <- apply(lentry$error, 1, sd)
      
    dataset[[i]]$best_model <- names(which.min(dataset[[i]]$errors))
    dataset[[i]]$best_model_error <- dataset[[i]]$errors[which.min(dataset[[i]]$errors)]

      
#     dataset[[i]]$errors <- rowMeans(lentry$mase_err)
  }
#   attr(dataset, "avg_snaive_errors") <- avg_snaive_errors
  dataset
}

#find best model
predict_interval_tps <- function(db_in, db_out, clamp_zero = FALSE) {
    
    for (i in 1:length(db_in)) {
    
      ff <- db_out[[i]]$ff
      ensemble = apply(ff, 2, mean)
      names(ensemble) = 'ensemble'
        
      ff <- rbind(ff, ensemble)
      colnames(ff) = NULL
        
      db_out[[i]]$ff = ff
      
      best_model = db_in[[i]]$best_model
      sderror = db_in[[i]]$sderror[best_model]
      radius = sderror * sqrt(1:db_in[[i]]$h)
        
      upper <- db_out[[i]]$ff[best_model,][1:length(radius)] + radius
      lower <- db_out[[i]]$ff[best_model,][1:length(radius)] - radius
        
      db_out[[i]]$upper <- ifelse(clamp_zero & upper < 0, 0, upper)
      db_out[[i]]$lower <- ifelse(clamp_zero & lower < 0, 0, lower)
      
      db_out[[i]]$best_model = best_model
    }
    
    db_out
    
  }