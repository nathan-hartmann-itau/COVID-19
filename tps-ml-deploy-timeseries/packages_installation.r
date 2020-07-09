r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
Sys.setenv(TAR = "/bin/tar")

packs = c(
    'repr'
    , 'remotes'
    , 'pacman'
    , 'tidyverse', 'magrittr', 'forecast', 'reticulate', 'tsibble', 'lubridate'
    , 'foreach', 'doMC', 'rlist', 'grid'
)

install.packages(packs)

#remotes::install_github("pmontman/tsfeatures")
remotes::install_github("pmontman/customxgboost")
remotes::install_github("robjhyndman/M4metalearning")