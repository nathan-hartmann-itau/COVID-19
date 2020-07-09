r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

packs = c(
    'tidyverse', 'magrittr', 'forecast', 'reticulate', 'tsibble', 'lubridate'
    , 'rjags'
    , 'matrixStats', 'mcmcplots'
    , 'foreach', 'doMC', 'repr', 'rlist', 'grid'
)

if(pacman::p_load(char = packs)){
    print('Todos os pacotes foram carregados!')
}

cores = 5

# session <- sagemaker$Session()
# role_arn <- sagemaker$get_execution_role()
# bucket = "todospelasaude-lake-raw-prod"

#bucket_in = "todospelasaude-lake-raw-prod"
#prefix_in = 'ExternalSupplier/Covid19BR/Full/Files/cases-brazil-states'
#object = 'cases-brazil-states.csv'
#s3_path = s3(bucket_in, prefix_in, object)

system("aws s3 cp s3://todospelasaude-lake-raw-prod/ExternalSupplier/Covid19BR/Full/Files/cases-brazil-states/cases-brazil-states.csv ./", intern=TRUE)
covid19 = read_csv("./cases-brazil-states.csv", guess_max = 10000)
#covid19 = read_s3(s3_path, delim = ",", col_names = TRUE, guess_max = 10000)

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
br_pop = read_csv("./ibge_population.csv", guess_max = 10000)
br_pop = br_pop %>% group_by(uf) %>% summarise(pop = sum(estimated_population)) %>% ungroup()
br_pop = br_pop %>% bind_rows(data.frame(uf = 'BR', pop = sum(br_pop$pop)))

target = 'n'
# target = 'd'

output_poisson = data.frame()

registerDoMC(cores)

### Poisson (UFMG)

poisson <- foreach(s = 1:dim(uf)[1]) %dopar% {
    source(paste0(getwd(), "/scripts/R/jags_poisson.R"))

    i = 4  # (2: confirmed, 3: deaths, 4: new confirmed, 5: new deaths)
    L = 300
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

    pop <- br_pop$pop[which(br_pop$uf == uf$state[s])]

    t = dim(Y)[1]

    # use static to provide initial values
    params = c("a", "b", "c", "f", "yfut", "mu")
    nc = 1  # 3
    nb = 90000  # 5e4
    thin = 10
    ni = 10000  # 5e4
    data_jags = list(y = Y[[i]], t = t, L = L)
    mod = try(jags.model(textConnection(mod_string_new), data = data_jags, n.chains = nc,
                         n.adapt = nb, quiet = TRUE))
    try(update(mod, n.iter = ni, progress.bar = "none"))
    mod_sim = try(coda.samples(model = mod, variable.names = params, n.iter = ni,
                               thin = thin, progress.bar = "none"))

    if (class(mod_sim) != "try-error" && class(mod) != "try-error") {

        mod_chain = as.data.frame(do.call(rbind, mod_sim))

        a_pos = "a"
        b_pos = "b"
        c_pos = "c"
        f_pos = "f"
        mu_pos = paste0("mu[", 1:(t + L), "]")
        yfut_pos = paste0("yfut[", 1:L, "]")
        L0 = 15

        mod_chain_y = as.matrix(mod_chain[yfut_pos])
        mod_chain_cumy = rowCumsums(mod_chain_y) + Y[[2]][t]


        ### list output
        df_predict <- data.frame(date = as.Date((max(Y$date) + 1):(max(Y$date) +
                                                                       L0), origin = "1970-01-01"), q25 = colQuantiles(mod_chain_cumy[, 1:L0],
                                                                                                                       prob = 0.025), med = colQuantiles(mod_chain_cumy[, 1:L0], prob = 0.5),
                                 q975 = colQuantiles(mod_chain_cumy[, 1:L0], prob = 0.975), m = colMeans(mod_chain_cumy[,
                                                                                                                        1:L0]))
        row.names(df_predict) <- NULL

        lt_predict <- lt_summary <- NULL

        # longterm
        L0 = 200

        # acha a curva de quantil
        lowquant <- colQuantiles(mod_chain_y[, 1:L0], prob = 0.025)
        medquant <- colQuantiles(mod_chain_y[, 1:L0], prob = 0.5)
        highquant <- colQuantiles(mod_chain_y[, 1:L0], prob = 0.975)

        NTC25 = sum(lowquant) + Y[[2]][t]
        NTC500 = sum(medquant) + Y[[2]][t]
        NTC975 = sum(highquant) + Y[[2]][t]


        ## flag
        cm <- pop * 0.025
        ch <- pop * 0.03
        flag <- 0  #tudo bem
        {
            if (NTC500 > cm)
                flag <- 2  #nao plotar
            else {
                if (NTC975 > ch) {
                    flag <- 1
                    NTC25 <- NTC975 <- NULL
                }
            }
        }  #plotar so mediana

        # vetor de data futuras e pega a posicao do maximo do percentil 25.
        dat.vec <- as.Date((max(Y$date) + 1):(max(Y$date) + L0), origin = "1970-01-01")
        dat.full <- c(Y[[1]], dat.vec)


        Dat25 <- Dat500 <- Dat975 <- NULL
        dat.low.end <- dat.med.end <- dat.high.end <- NULL

        mod_chain_mu = as.matrix(mod_chain[mu_pos])
        mu50 <- apply(mod_chain_mu, 2, quantile, probs = 0.5)
        Dat500 <- dat.full[which.max(mu50[1:(t + L0)])]

        q <- 0.99
        med.cum <- c(medquant[1] + Y[[2]][t], medquant[2:length(medquant)])
        med.cum <- colCumsums(as.matrix(med.cum))
        med.cum <- med.cum/med.cum[length(med.cum)]
        med.end <- which(med.cum - q > 0)[1]
        dat.med.end <- dat.vec[med.end]

        if (flag == 0) {
            # definicao do pico usando a curva das medias
            mu25 <- apply(mod_chain_mu, 2, quantile, probs = 0.025)
            mu975 <- apply(mod_chain_mu, 2, quantile, probs = 0.975)

            posMax.q25 <- which.max(mu25[1:(t + L0)])
            aux <- mu975 - mu25[posMax.q25]
            aux2 <- aux[posMax.q25:(t + L0)]
            val <- min(aux2[aux2 > 0])
            dat.max <- which(aux == val)

            aux <- mu975 - mu25[posMax.q25]
            aux2 <- aux[1:posMax.q25]
            val <- min(aux2[aux2 > 0])
            dat.min <- which(aux == val)

            Dat25 <- dat.full[dat.min]
            Dat975 <- dat.full[dat.max]

            # calcula o fim da pandemia
            low.cum <- c(lowquant[1] + Y[[2]][t], lowquant[2:length(lowquant)])
            low.cum <- colCumsums(as.matrix(low.cum))
            low.cum <- low.cum/low.cum[length(low.cum)]
            low.end <- which(low.cum - q > 0)[1]
            dat.low.end <- dat.vec[low.end]

            high.cum <- c(highquant[1] + Y[[2]][t], highquant[2:length(highquant)])
            high.cum <- colCumsums(as.matrix(high.cum))
            high.cum <- high.cum/high.cum[length(high.cum)]
            high.end <- which(high.cum - q > 0)[1]
            dat.high.end <- dat.vec[high.end]
        }

        lt_predict <- data.frame(date = dat.vec, q25 = lowquant, med = medquant,
                                 q975 = highquant, m = colMeans(mod_chain_y[, 1:L0]))
        row.names(lt_predict) <- NULL

        lt_summary <- list(NTC25 = NTC25, NTC500 = NTC500, NTC975 = NTC975, high.dat.low = Dat25,
                           high.dat.med = Dat500, high.dat.upper = Dat975, end.dat.low = dat.low.end,
                           end.dat.med = dat.med.end, end.dat.upper = dat.high.end)

        muplot <- data.frame(date = dat.full, mu = mu50[1:(t + L0)])
        list_out <- list(df_predict = df_predict, lt_predict = lt_predict, lt_summary = lt_summary,
                         mu_plot = muplot, flag = flag, st = estado, convergence = 1)

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

out_poisson = data.frame()
for (s in uf$state){
    idx = s
    lentry = rlist::list.filter(poisson, st == idx)[[1]]
    if(lentry$convergence == 1){
        out = lentry$df_predict %>% select(date, m) %>% rename(date_pred = date, pred = m) %>% head(15)
        out$id = idx
        out$fold = max(covid19$date)
        out_poisson = bind_rows(out_poisson, out)
    }
}

df_tmp = tryCatch({
    system('aws s3 cp s3://todospelasaude-lake-deploys/predictions/states/poisson__num_casos.csv ./', intern=TRUE)
    read_csv('./poisson__num_casos.csv', guess_max = 10000)
    },
    error = function(err) {
        return(NA)
    })

if(is.na(df_tmp)){
    write_csv(out_poisson, './poisson__num_casos.csv')

} else {
    write_csv(df_tmp %>% bind_rows(out_poisson), './poisson__num_casos.csv')

}

system('aws s3 cp ./poisson__num_casos.csv s3://todospelasaude-lake-deploys/predictions/states/poisson__num_casos.csv', intern=TRUE)