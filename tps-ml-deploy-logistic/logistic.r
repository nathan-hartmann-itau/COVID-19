packs = c(
    'tidyverse', 'magrittr', 'forecast', 'reticulate', 'tsibble', 'lubridate'
    , 'foreach', 'doMC', 'repr', 'rlist', 'grid'
)

if(pacman::p_load(char = packs)){
    print('Todos os pacotes foram carregados!')
}

cores = 5

system("aws s3 cp s3://todospelasaude-lake-raw-prod/ExternalSupplier/Covid19BR/Full/Files/cases-brazil-states/cases-brazil-states.csv ./", intern=TRUE)
covid19 = read_csv("./cases-brazil-states.csv")

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

system("aws s3 cp s3://todospelasaude-lake-deploys/manual_inputs/ibge_population.csv ./", intern=TRUE)
br_pop = read_csv("./ibge_population.csv")
br_pop = br_pop %>% group_by(uf) %>% summarise(pop = sum(estimated_population)) %>% ungroup()
br_pop = br_pop %>% bind_rows(data.frame(uf = 'BR', pop = sum(br_pop$pop)))

target = 'n'
# target = 'd'

registerDoMC(cores)
    
### Modelo logístico

logistic <- foreach(s = 1:dim(uf)[1]) %dopar% {
    estado = uf$state[s]

    target_diario = paste0(target, '_new')

    h = 3*30
    b = 200

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

    set.seed(1234)

    db_fpl = data.frame(
        y = Y %>% pull((!!target)),
        t = 1:dim(Y)[1]
    )
    res = tryCatch(
        getInitial(as.formula(y ~ SSfpl(t, A, B, xmid, scal)), data = db_fpl),
        error = function(err) {
            return(NA)
        })
    if(is.na(res)){
        model = as.formula(y ~ SSlogis(t, Asym, xmid, scal))

    } else {
        model = as.formula(y ~ SSfpl(t, A, B, xmid, scal))

    }

    sim <- bld.mbb.bootstrap(Y %>% pull((!!target_diario)) %>% + 10 %>% log, b) %>%
      as.data.frame()

    b_sim = sim %>% 
        rename_all(funs(make.names(1:b) %>% str_replace_all(c("X" = "B")))) %>% 
        mutate_all(~cumsum( ifelse(exp(.) - 10 < 0, 0, exp(.) - 10) )) %>% 
        gather(boot, y) %>% 
        group_by(boot) %>% 
        mutate(t = 1:n()) %>% 
        ungroup()

    candidates = b_sim %>% pull(boot) %>% unique()

    last_t = max(b_sim$t)

    newdata = data.frame(t = seq(last_t + 1,last_t + h, 1))
    db_boot = data.frame()

    picos = numeric()
    i = 1
    while(length(picos[!is.na(picos)]) <= 100){
        res = tryCatch({
        b_fit <- nls(
                model,
                data = b_sim %>% filter(boot == candidates[i]),
                #trace = TRUE,
                algorithm = "port",
                #start = newstart,
                control = nls.control(maxiter = 250, minFactor = 1/(1024 * 4))
            )
            }, error = function(err) {
                return(NA)
            })
        if(class(res) != 'nls'){
            picos = c(picos, NA)
        } else {
            db_forecast = data.frame(
                pred = predict(b_fit, newdata, type = "response"),
                t = newdata$t,
                boot = candidates[i]
            )
            db_boot = rbind(db_boot, db_forecast)
            picos = c(picos, coef(b_fit)['xmid'])
        }
        i = i + 1
        if  ( i == b ) break
    }

    picos = as.numeric(picos[!is.na(picos)])

    if (length(picos) >= 100) {

        db_boot = db_boot %>% 
            select(-one_of('boot')) %>%  
            mutate(t = as.character(t)) %>%
            group_by(t) %>% 
            summarise(media = mean(pred), lwr = quantile(pred, .025), upr = quantile(pred, .975)) %>% 
            mutate(t = as.numeric(t)) %>% 
            ungroup() %>% 
            rename(pred = media) %>% 
            arrange(t)

        data_inicio = Y %$% date %>% min()

        picos_potenciais = round(c(quantile(picos, probs = 0.025), mean(picos), quantile(picos, probs = 0.975)))

        db_plot = bind_rows(
            data.frame(
                y = Y %>% pull((!!target)),
                t = 1:(length(dim(Y)[1]))
            )
            , db_boot
        )

        db_plot = db_plot %>% 
            mutate(
                y_comb = round(ifelse(is.na(y), pred, y))

                , y_comb_diario = difference(y_comb)
                , y_comb_diario = ifelse(is.na(y_comb_diario), y_comb, y_comb_diario)
                , y_comb_diario = ifelse(y_comb_diario < 0, NA, y_comb_diario)

                , y_diario = difference(y)
                , y_diario = ifelse(is.na(y_diario), y, y_diario)

                , pred_diario = difference(y_comb)
                , pred_diario = ifelse(is.na(pred_diario), y_comb, pred_diario)
                , pred_diario = ifelse(is.na(pred), NA, pred_diario)
                , pred_diario = ifelse(pred_diario < 0, NA, pred_diario)

                , lwr_comb = round(ifelse(is.na(y), lwr, y))
                , lwr_diario = difference(lwr_comb)
                , lwr_diario = ifelse(is.na(lwr_diario), y_comb, lwr_diario)
                , lwr_diario = ifelse(is.na(pred), NA, lwr_diario)
                , lwr_diario = ifelse(lwr_diario < 0, NA, lwr_diario)

                , upr_comb = round(ifelse(is.na(y), upr, y))
                , upr_diario = difference(upr_comb)
                , upr_diario = ifelse(is.na(upr_diario), y_comb, upr_diario)
                , upr_diario = ifelse(is.na(pred), NA, upr_diario)
                , upr_diario = ifelse(upr_diario < 0, NA, upr_diario)

                , date = data_inicio + 0:(nrow(db_plot) - 1)
            )

        df_predict = db_plot %>%
            filter(is.na(y)) %>%
            select(date, pred, lwr, upr) %>% 
            rename(m = pred, q25 = lwr, q975 = upr)

        lt_predict = db_plot %>%
            filter(is.na(y)) %>%
            select(date, pred_diario, lwr_diario, upr_diario) %>% 
            rename(m = pred_diario, q25 = lwr_diario, q975 = upr_diario)

        list_out = list(
            df_predict = df_predict,
            lt_predict = lt_predict,
            high.dat.low = data_inicio + picos_potenciais[1],
            high.dat.m = data_inicio + picos_potenciais[2],
            high.dat.upper = data_inicio + picos_potenciais[3],
            st = estado,
            convergence = 1,
            formula = model
        )

        list_out

    } else {
        list_out = list(
            st = estado,
            convergence = 0
#             print(paste0("ERROR:", uf$state[s]))
        )

        list_out
    }
}

out_logistic = data.frame()
for (s in uf$state){
    idx = s
    lentry = rlist::list.filter(logistic, st == idx)[[1]]
    if(lentry$convergence == 1){
        out = lentry$df_predict %>% select(date, m) %>% rename(date_pred = date, pred = m) %>% head(15)
        out$id = idx
        out$fold = max(covid19$date)
        out_logistic = bind_rows(out_logistic, out)
    }
}

df_tmp = tryCatch({
    system('aws s3 cp s3://todospelasaude-lake-deploys/predictions/states/logistic__num_casos.csv ./', intern=TRUE)
    read_csv('./logistic__num_casos.csv', guess_max = 10000)
    },
    error = function(err) {
        return(NA)
    })

if(is.na(df_tmp)){
    write_csv(out_logistic, './logistic__num_casos.csv')

} else {
    write_csv(df_tmp %>% bind_rows(out_logistic), './logistic__num_casos.csv')

}

system('aws s3 cp ./logistic__num_casos.csv s3://todospelasaude-lake-deploys/predictions/states/logistic__num_casos.csv', intern=TRUE)