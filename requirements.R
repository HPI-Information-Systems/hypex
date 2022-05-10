# CRAN dependencies
packages <- c("formatR", "pcalg")

# Non-CRAN dependencies
bioc.packages <- c("graph", "RBGL")

not_installed <- function(pkg) {
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    return(length(new.pkg) > 0)
}

# Use as CRAN mirror
repos <- "http://cran.r-project.org"

if (!require("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = repos, ask = FALSE)
}

# Make sure to install the Non-CRAN dependencies first, as pcalg depends on it
print("Now installing non-CRAN packages")
for (pkg in bioc.packages) {
    if (not_installed(pkg)) {
        print(paste0("Now installing ", pkg))
        BiocManager::install(pkg, update = FALSE)
        sapply(pkg, require, character.only = TRUE)
        print(paste0("Successfully installed ", pkg))
    }
}
print("Successfully installed non-CRAN packages")

print("Now installing CRAN packages")
for (pkg in packages) {
    if (not_installed(pkg)) {
        print(paste0("Successfully installed ", pkg))
        install.packages(pkg, repos = repos, ask = FALSE)
        sapply(pkg, require, character.only = TRUE)
        print(paste0("Successfully installed ", pkg))
    }
}
print("Successfully installed CRAN packages")
