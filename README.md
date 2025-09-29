# ETL Pipeline Assignment

A Python-based ETL pipeline for processing project, subject, and sample data from CSV files, cleaning and merging them,
and exporting results in Parquet and CSV formats.

## Requirements

- Docker
- CSV files in `data/` directory:
    - projects.csv
    - subjects.csv
    - samples.csv

## Option 1: Download and Run from Docker Hub

```bash
docker pull rhigaze/etl-processor:latest
docker run --rm  -e PROJECTS_DATA_CSV_FILE=project_study_cohort.csv -e SUBJECTS_DATA_CSV_FILE=subject_samples.csv -e SAMPLES_DATA_CSV_FILE=sample_run_results.csv -e OUTPUT_CSV=output.csv -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output rhigaze/etl-processor
```
Note: Replace the environment variable values with your actual file names if needed.


## Option 2:  Clone from GitHub, Build, and Run
  

**Note:** Building the Docker image may take a minute, while pulling the pre-built image (Option 1) is faster.
### Clone the Repository

```bash  
git clone https://github.com/rhigaze/ETL.git

cd ETL
```


## Build Docker Image

```bash
docker build -t etl-processor .
```

## Run ETL Pipeline

```bash 
docker run --rm  -e PROJECTS_DATA_CSV_FILE=project_study_cohort.csv -e SUBJECTS_DATA_CSV_FILE=subject_samples.csv -e SAMPLES_DATA_CSV_FILE=sample_run_results.csv -e OUTPUT_CSV=output.csv -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output etl-processor
```

Note: Replace the environment variable values with your actual file names if needed.

## Output

After running the pipeline, the following files will be generated in the `output/` directory:

- `merged_data.parquet`: Contains the full merged and cleaned dataset in Parquet format.
- `statistics_report.csv`: Summary statistics report in CSV format.
- `anomalies_report.csv`: Anomalies detected during processing, in CSV format.

## ---------------------------------------------------------------------------
## Agent
The Data Analysis Agent is a Python-based AI assistant designed to help analysts interact with a dataset using natural language. Built with LangChain and GPT-4o-mini, the agent can:
Search the dataset with flexible queries and return all matching rows.
Monitor data for anomalies such as extreme values or low-quality samples.
Visualize data by generating histograms of any column, returned as images ready for display.

# Requirements

```bash
pip3 install fastparquet==2024.11.0 pandas==2.3.2 matplotlib==3.10.6 langchain==0.3.27 langchain-openai==0.3.33 langchain-experimental==0.3.4 langchain-core==0.3.76 openai==1.109.1 tabulate==0.9.0
```

# Run agent
```bash
python3 agents.py --file "output/merged_data.parquet"
```