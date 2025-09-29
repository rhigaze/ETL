# etl_pipeline.py

import os
import re
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check, errors

VALID_STATUS = ["Finished", "Running", "Failed"]
# Define file paths from environment variables or use defaults
output_dir = os.environ.get("OUTPUT_DIR", "output")
data_dir = os.environ.get("DATA_DIR", "data")
anomaly_dir = os.environ.get("ANOMALY_DIR", f"{output_dir}/anomaly")
# input files
projects_csv_file = f"{data_dir}/{os.environ.get('PROJECTS_DATA_CSV_FILE', 'project_study_cohort.csv')}"   # default if not passed
subjects_csv_file = f"{data_dir}/{os.environ.get('SUBJECTS_DATA_CSV_FILE', 'subject_samples.csv')}"   # default if not passed
samples_csv_file = f"{data_dir}/{os.environ.get('SAMPLES_DATA_CSV_FILE', 'sample_run_results.csv')}"   # default if not passed
# output files
merged_data_file = f"{output_dir}/{os.environ.get('OUTPUT_PARQUET_FILE', 'merged_data.parquet')}"   # default if not passed
summary_report_file = f"{output_dir}/{os.environ.get('SUMMARY_OUTPUT_CSV', 'summary_statistics.csv')}"   # default if not passed


# Define schema
schema = DataFrameSchema(
    {
        "sample_id": Column(int, Check.greater_than(0)),
        "cancer_detected": Column(
            str,
            checks=Check.isin(["Yes", "No"]),
            nullable=True,
            description="Cancer detection flag"
        ),
        "detection_value": Column(float, Check.in_range(0, 1),coerce=True, nullable=True),
        "sample_quality": Column(float, Check.in_range(0, 1), coerce=True, nullable=True),
        "sample_quality_minimum_threshold": Column(float),
        "sample_status": Column(
            str,
            checks=Check.isin(VALID_STATUS+["Unknown"]),
            nullable=True,
        ),
        "fail_reason": Column(str, nullable=True),
        "date_of_run": Column(object, nullable=True),
        "project_code": Column(str, nullable=False),
        "study_code": Column(str, nullable=False),
        "study_cohort_code": Column(str, nullable=False),
        "subject_id": Column(int, Check.greater_than(0)),
        "type": Column(str, nullable=False),
        "project_name": Column(str, nullable=False),
        "study_name": Column(str, nullable=False),
        "study_cohort_name": Column(str, nullable=False),
        "disease_name": Column(str, nullable=False),
        "project_manager_name": Column(str, nullable=True),
    },
    coerce=True,
    strict=True,   # Disallow unexpected columns
    name="CancerDetectionSchema",
)

def clean_column_name(name:str) -> str:
    """
    Cleans a single column name:
    - Strips spaces
    - Converts to lowercase
    - Removes text in parentheses
    - Replaces non-alphanumeric characters with underscore
    """
    name = name.strip().lower()
    # Remove text inside parentheses (including the parentheses)
    name = re.sub(r"\(.*?\)", "", name)
    # Replace any remaining non-alphanumeric characters with underscore
    name = re.sub(r"[^\w]", "_", name)
    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name

def save_anomalies(df, name):
    """Save anomalies with timestamp"""
    os.makedirs(anomaly_dir, exist_ok=True)
    path = f"{anomaly_dir}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"Anomalies saved to {path} - rows : {df.shape[0]}")

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame|None:
    """
    Clean and validate the DataFrame using Pandera schema. Logs anomalies to CSV files.
    """
    # Save anomalies (missing project_name)
    anomalies = df[df["project_name"].isna()]
    if not anomalies.empty:
        save_anomalies(df=anomalies, name="samples_with_empty_project")

    print(f"Data before cleaning: {df.shape[0]} rows")

    # Normalize column names
    df.columns = [clean_column_name(col) for col in df.columns]

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop unsupported samples
    df = df[~df["project_name"].isna()]

    # Fix invalid sample_status → "Unknown" and log anomalies
    df_with_invalid_status = df[~df["sample_status"].isin(VALID_STATUS)]
    df.loc[~df["sample_status"].isin(VALID_STATUS), "sample_status"] = "Unknown"
    if not df_with_invalid_status.empty:
        save_anomalies(df=anomalies, name="samples_with_invalid_status")

    # Clean numeric columns, coerce errors to NaN
    numeric_cols = ["detection_value", "sample_quality", "sample_quality_minimum_threshold"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    # Check schema validation
    try:
        validated = schema.validate(df)
        print(f"Schema validation passed: {validated.shape[0]} rows")
        return validated
    except errors.SchemaErrors as err:
        print(f"Schema validation failed with {len(err.failure_cases)} issues")
        save_anomalies(err.failure_cases, "schema_anomalies")
        raise err


def extract_data(csv_file: str, subset:list=None) -> pd.DataFrame|None:
    """
           Read a CSV file into a DataFrame, drop duplicates and rows with missing values in subset.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Error: {csv_file} not found!")

    print(f"Extracting data from {csv_file}...")

    df = pd.read_csv(csv_file).drop_duplicates()
    if subset: df = df.dropna(subset=subset)

    print(f"{csv_file} - data extraction completed successfully. Total row : {df.shape[0]}")
    return df

def transform_data(projects: pd.DataFrame, subjects: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """
     Merge and clean project, subject, and sample data. Handles anomalies and validates types.
     Returns:
         pd.DataFrame: Cleaned transformed DataFrame.
     """
    print(f"Transform data ...")
    # Merge samples with subjects
    df = samples.merge(subjects,
                       on="sample_id",
                       how="inner",
                       validate="1:1")

    # Merge with projects
    df = df.merge(projects,
                  on=["project_code", "study_code", "study_cohort_code"],
                  how="left",
                  validate="m:1")

    # Clean and validate
    valid_df = clean_and_validate(df)
    print(f"Data transforming completed successfully. Total row : {valid_df.shape[0]}")
    return valid_df

def load_data(df: pd.DataFrame, output_file:str):
    """
      Save the DataFrame to a Parquet file in the specified output directory.
    """
    print(f"Loading data to parquet file : 'output_csv_file'...")
    df.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"Data Loading completed successfully")

def create_summary(df: pd.DataFrame, output_file:str):
    """
    Create a summary report from the DataFrame and save it as a CSV file.
    """
    summary = (
        df.groupby(["project_name", "study_name", "study_cohort_name" ,"project_code", "study_code", "study_cohort_code"])
        .agg(
            samples_detected=("sample_id", "count"),
            sample_with_status_finished=("sample_status", lambda x: x.eq("Finished").sum()),
            finished_pct=("sample_status", lambda x: (x.eq("Finished").mean() * 100)),
            lowest_detection_value=("detection_value", "min"),
        )
        .reset_index()
    )
    summary.to_csv(output_file, index=False)
    print("Summary Data completed successfully.")

def run_etl_pipline():
    """
      Run the full ETL pipeline: extract, transform, load, and summarize data
      .
    """
    # Extract
    projects_df = extract_data(csv_file=projects_csv_file,
                               subset=["project_code", "study_code", "study_cohort_code"])
    subjects_df = extract_data(csv_file=subjects_csv_file,
                               subset=["project_code", "study_code", "study_cohort_code","sample_id", "subject_id"])
    samples_df = extract_data(csv_file=samples_csv_file,
                              subset=["sample_id"])

    # Transform
    df = transform_data(projects=projects_df,
                        subjects=subjects_df,
                        samples=samples_df)

    # Load
    load_data(df=df ,
              output_file=merged_data_file)

    print("ETL pipeline completed successfully. \n")

    # Summary for each project
    create_summary(df=df,
                   output_file=summary_report_file)


if __name__ == "__main__":
    run_etl_pipline()
