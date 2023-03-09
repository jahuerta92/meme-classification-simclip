FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04

# Install git
RUN apt-get update
RUN apt-get install -y git

# Install curl
RUN apt-get update
RUN apt-get -y install curl
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash

# Install nodejs
RUN apt-get install -y nodejs

# Other ubuntu packages
RUN apt-get update
RUN apt-get -y install zsh wget sudo nano screen zip

# git global configuration
RUN git config --global pull.rebase true
RUN git config --global rebase.autoStash true 

# Install python
RUN apt-get update
RUN apt-get -y install python3 python3-pip

# Install tesseract
RUN apt update
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev

# Install jupyterlab
RUN pip3 install --upgrade jupyter
RUN pip3 install --upgrade jupyterlab
RUN pip3 install jupyterlab-requirements
RUN pip3 install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user
RUN jupyter nbextension enable collapsible_headings/main --user

### Jupyterlab extensions
RUN pip3 install --upgrade jupyterlab-git nbdime aquirdturtle_collapsible_headings jupyterlab_widgets jupyterlab-quickopen ipykernel \
                           jupyterlab_nvdashboard jupyterlab-logout jupyterlab-system-monitor jupyterlab-topbar \
                           jupyterlab_theme_hale jupyterlab_theme_solarized_dark nbresuse \
                           jupyter-lsp jupyterlab-drawio jupyter-dash jupyterlab_code_formatter black isort jupyterlab_latex \
                           xeus-python theme-darcula jupyterlab_materialdarker lckr-jupyterlab-variableinspector

# PYTHON PACKAGES with pip
RUN pip3 install --upgrade scikit-learn fastgpu pandas transformers tokenizers datasets \
                           pymongo emoji python-dotenv colored pytesseract \
                           wandb spacy sentencepiece einops pytorch-lightning sentencepiece\
                           nltk seaborn torch torchvision torchaudio \
                           --extra-index-url https://download.pytorch.org/whl/cu116

# Jupyterlab lsp
RUN jupyter lab clean
RUN jupyter lab build --dev-build=False --minimize=False

# Environmental variables for wandb
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install Github CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	| gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
	| tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh

# Change the shell to bash
SHELL [ "/bin/bash", "--login", "-c" ]

EXPOSE 8888

# Defining ARG variables passed from docker-compose
ARG UID
ARG GID
ARG USER

RUN echo ${UID}
RUN echo ${GID}
RUN echo ${USER}

RUN addgroup --gid $GID $USER
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USER

# Giving root permissions to $USER
RUN adduser $USER sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $USER

WORKDIR /home/$USER/work

ENV SHELL=/bin/zsh

# Pretty zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

CMD ["jupyter","lab","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root","--ContentsManager.allow_hidden=True"]
