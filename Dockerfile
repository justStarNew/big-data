FROM jupyter/all-spark-notebook

MAINTAINER Pierre Navaro <pierre.navaro@univ-rennes1.fr>

USER root

COPY . ${HOME}

RUN chown -R ${NB_USER} ${HOME}

USER $NB_USER

RUN conda env update -n base --quiet -f environment.yml && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR
