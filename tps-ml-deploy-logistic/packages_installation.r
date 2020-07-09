r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

packs = c(
    'repr'
    , 'remotes'
    , 'pacman'
    , 'tidyverse', 'magrittr', 'forecast', 'reticulate', 'tsibble', 'lubridate'
    , 'foreach', 'doMC', 'rlist', 'grid'
    , 'sagemaker'
)

install.packages(packs)