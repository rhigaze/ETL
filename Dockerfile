# Stage 1: Builder
FROM python:3.10-alpine AS builder

WORKDIR /app

# Install build dependencies and git
RUN apk add --no-cache \
    git \
    build-base

# Copy and install Python packages into /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime
FROM python:3.10-alpine

WORKDIR /app

# Copy only the installed Python packages from builder
COPY --from=builder /install /usr/local

COPY data_pipeline.py .

# Create folders and set environment variables
ENV OUTPUT_DIR=output \
    DATA_DIR=data \
    PROJECTS_DATA_CSV_FILE=project_study_cohort.csv \
    SUBJECTS_DATA_CSV_FILE=ubject_samples.csv \
    SAMPLES_DATA_CSV_FILE=sample_run_results.csv \
    OUTPUT_PARQUET_FILE=merged_data.parquet\
    SUMMARY_OUTPUT_CSV=statistics_report.csv
RUN mkdir -p $OUTPUT_DIR $DATA_DIR

# Run the ETL script
ENTRYPOINT ["python3", "data_pipeline.py"]
