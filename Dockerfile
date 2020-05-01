FROM jupyter/tensorflow-notebook

USER jovyan

WORKDIR /home/jovyan/work/debruiteur

COPY ./ ./

RUN pip install opencv-python
RUN pip install ./debruiteur --user

ENTRYPOINT ["jupyter", "lab", "--NotebookApp.token=''", "--ip=0.0.0.0", "--allow-root"]