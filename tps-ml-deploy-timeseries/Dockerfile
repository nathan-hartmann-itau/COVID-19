FROM continuumio/anaconda3:latest

WORKDIR /app

#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#RUN unzip awscliv2.zip
#RUN sudo ./aws/install
RUN conda create -y -n r_env r-essentials r-base
SHELL ["conda", "run", "-n", "r_env", "/bin/bash", "-c"]

RUN conda install -y -c conda-forge jags

COPY ./packages_installation.r ./

RUN Rscript packages_installation.r

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY ./ ./
ENTRYPOINT ["conda", "run", "-n", "r_env", "Rscript", "time_series.r"]