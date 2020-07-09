r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

packs = c(
    'tidyverse', 'magrittr', 'forecast', 'reticulate', 'tsibble', 'lubridate'
    , 'M4metalearning'
    , 'foreach', 'doMC', 'repr', 'rlist', 'grid'
)

if(pacman::p_load(char = packs)){
    print('Todos os pacotes foram carregados!')
}

cores = 5

system("aws s3 cp s3://todospelasaude-lake-raw-prod/ExternalSupplier/Covid19BR/Full/Files/cases-brazil-states/cases-brazil-states.csv ./", intern=TRUE)
covid19 = read_csv("./cases-brazil-states.csv", guess_max = 10000)

covid19 <- covid19 %>%
    rename(n = totalCasesMS, d = deathsMS) %>% 
    mutate(date = as.Date(date), state = ifelse(state == 'TOTAL', 'BR', state)) %>% 
    select(date, n, d, state) %>%
    arrange(state, date) %>% 
    group_by(state) %>%
    ungroup() %>%
    filter(date >= "2020-01-23")
    # filter(date <= today('America/Sao_Paulo'))

uf <- distinct(covid19, state)

target = 'n'
# target = 'd'

registerDoMC(cores)

ts_list = list()
h = 15
freq = 7

for(s in 1:dim(uf)[1]){
    estado = uf$state[s]
    Y <- covid19 %>% filter(state == estado) %>% mutate(n_new = n - lag(n, default = 0), 
                                                        d_new = d - lag(d, default = 0)) %>% select(date, n, d, n_new, d_new, state) %>% 
        arrange(date)

    while (any(Y$n_new < 0)) {
        pos <- which(Y$n_new < 0)
        for (j in pos) {
            Y$n_new[j - 1] = Y$n_new[j] + Y$n_new[j - 1]
            Y$n_new[j] = 0
            Y$n[j - 1] = Y$n[j]
        }
    }

    ts_list[[s]] = list(
        st = estado
        , x = (ts(Y %>% pull((!!target)), frequency = freq))
        , h = as.integer(h)
        , data_inicio = min(Y$date)
        , data_fim = max(Y$date)
    )
}

source(paste0(getwd(), '/scripts/R/meta_functions.R'))

#create the training set using temporal holdout
train <- temp_holdout(ts_list)

#calculate the forecasts of each method in the pool
train <- calc_forecasts(train, forec_methods(), n.cores = cores)

#performance evaluation
train = calc_errors_tps(train)

#final database scoring
time_series <- ts_list

#just calculate the forecast and features
time_series <- calc_forecasts(time_series, forec_methods(), n.cores = cores)

#interval prediction
time_series = predict_interval_tps(train, time_series, clamp_zero = TRUE)

out_time_series = data.frame()
for (s in uf$state){
    idx = s
    lentry = rlist::list.filter(time_series, st == idx)[[1]]

    out = lentry$ff %>% t %>% as.data.frame()
    out$date_pred = lentry$data_fim + 1:lentry$h
    out$id = idx
    out$fold = lentry$data_fim
    out_time_series = bind_rows(out_time_series, out)
}

target_out = 'num_casos'

# salvar modelos temporais no s3
ts_models = names(out_time_series)[!names(out_time_series) %in% c('date_pred', 'id', 'fold')]

for(model_out in ts_models){
    
    out_model = out_time_series %>%
        select(one_of(c(model_out, 'date_pred', 'id', 'fold'))) %>%
        rename(pred = !!(model_out))
    
    df_tmp = tryCatch({
        object = paste0('aws s3 cp s3://todospelasaude-lake-deploys/predictions/states/', model_out, '__', target_out, '.csv ./')
        system(object, intern=TRUE)
        read_csv(paste0('./', model_out, '__', target_out, '.csv'), guess_max = 10000)
        },
        error = function(err) {
            return(NA)
        })

    if(is.na(df_tmp)){
        write_csv(out_model, './out_model.csv')

    } else {
        write_csv(df_tmp %>% bind_rows(out_model), './out_model.csv')

    }
    object = paste0('aws s3 cp ./out_model.csv s3://todospelasaude-lake-deploys/predictions/states/', model_out, '__', target_out, '.csv')
    system(object, intern=TRUE)

}