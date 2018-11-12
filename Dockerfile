FROM jupyter/all-spark-notebook

MAINTAINER Pierre Navaro <pierre.navaro@univ-rennes1.fr>

USER root

COPY . ${HOME}

RUN chown -R ${NB_USER} ${HOME}


RUN conda env update -n base --quiet -f environment.yml && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR

USER $NB_USER

RUN  jupyter labextension install dask-labextension \
        jupyterlab_bokeh @jupyterlab/hub-extension \
        @jupyter-widgets/jupyterlab-manager
