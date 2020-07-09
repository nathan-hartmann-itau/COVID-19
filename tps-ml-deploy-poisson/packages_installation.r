r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

#system("conda install -y -c conda-forge jags", intern=TRUE)

Sys.setenv(PKG_CONFIG_PATH = "/usr/lib64")
Sys.getenv("PKG_CONFIG_PATH")

install.packages("rjags", dependencies = TRUE, keep_outputs = TRUE, configure.args="--enable-rpath")

packs = c(
    'repr'
    , 'remotes'
    , 'pacman'
    , 'tidyverse', 'magrittr', 'forecast', 'reticulate', 'tsibble', 'lubridate'
    , 'matrixStats', 'mcmcplots'
    , 'foreach', 'doMC', 'rlist', 'grid'
    , 'sagemaker'
)

install.packages(packs)