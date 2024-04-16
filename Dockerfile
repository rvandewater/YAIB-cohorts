FROM r-base:4.2.2
COPY /R /R
COPY config.yaml config.yaml
COPY /ricu-extensions /ricu-extensions
#RUN apt-install -y git
#RUN git clone https://github.com/rvandewater/YAIB-cohorts.git
RUN R -e "install.packages('renv', repos = c(CRAN = 'https://cloud.r-project.org'))"

#RUN mkdir -p renv
#COPY .Rprofile .Rprofile
#COPY renv/activate.R renv/activate.R
#COPY renv/settings.json renv/settings.json
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev libudunits2-dev
RUN cd /R
ENV RENV_PATHS_LIBRARY renv/library
#RUN cd /R && R --vanilla -s -e 'renv::restore()'
# utils package isn't installed automatically; demo packages need to be installed afterwards, as well as tidyverse
#RUN cd /R && R --vanilla -s -e 'renv::activate()'  \
#    && Rscript "extra_setup.R" \

#&& R --vanilla 'install.packages("ricu")'

#    R --vanilla 'install.packages(c("mimic.demo","eicu.demo"), repos="https://eth-mds.github.io/physionet-demo")' \
#    && R --vanilla 'install.packages("tidyverse")'

#RUN cd /R && Rscript "base_cohort.R" --src mimic_demo
#install.packages("units")
#
## same for the demo datasets
#install.packages("mimic.demo", repos="https://eth-mds.github.io/physionet-demo")
#install.packages("eicu.demo", repos="https://eth-mds.github.io/physionet-demo")

#RUN cd /R &&  \
#    Rscript setup_env.R