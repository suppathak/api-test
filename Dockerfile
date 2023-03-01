FROM jupyter/scipy-notebook

ENV MODEL_DIR =../aicoe-osc-demo/models/distilbert_mnli_pruned80
ENV MODEL_FILE=model.onnx

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY api.py ./api.py

#USER 1001
EXPOSE 8080

RUN python3 train.py
CMD ["python3", "api.py", "8080"]
