FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy project
COPY . .

# run full pipeline
CMD sh -c "\
python src/data/build_combined_dataset.py && \
python src/models/run_classification.py && \
python src/models/run_regression.py && \
python src/models/run_npml_pipeline.py && \
python src/visualization/generate_plots.py && \
python src/visualization/generate_npml_plots.py"