import io
import base64

import pandas as pd
import matplotlib.pyplot as plt

from langchain.tools import tool
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import SystemMessage
from langchain_experimental.tools.python.tool import PythonAstREPLTool

OPEN_AI_API_KEY = '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'


class DataAgentTools:
    # Agent to create reports and visualizations
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @tool
    def plot_data(self, column: str) -> str:
        """
        Create simple plots from DataFrame.
        Example: 'validated_df["detection_value"].hist()'
        """
        plt.clf()
        df[column].hist()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"<img src='data:image/png;base64,{img_b64}'/>"

    # Agent for search inside the data
    @tool
    def search_data_tool(self, query: str) :
        """Search the dataset for rows containing the specified keyword in a specific column."""
        try:
            df_res = self.df.query(query)
            if df_res.empty:
                return "No matching rows found."
            print(df_res.head(1).to_dict(orient="records"))
            return df_res.head(1).to_dict()# returns CSV string
        except Exception as e:
            return f"Error: {e}"

    # Agent for monitoring data quality and anomalies
    @tool
    def monitor_data(self) -> str:
            """
            Monitor data quality and detect anomalies in numeric columns.
            Returns a summary of missing values, outliers, and anomaly detection.
            """
            report = []
            # Check for missing values
            missing = self.df.isnull().sum()
            missing_cols = missing[missing > 0]
            if not missing_cols.empty:
                report.append(f"Missing values:\n{missing_cols.to_dict()}")
            else:
                report.append("No missing values detected.")

            # Check for duplicate rows
            dup_count = self.df.duplicated().sum()
            if dup_count > 0:
                report.append(f"Duplicate rows: {dup_count}")
            else:
                report.append("No duplicate rows detected.")

            # Outlier and anomaly detection for numeric columns
            numeric_df = self.df.select_dtypes(include="number")
            if numeric_df.empty:
                report.append("No numeric columns to monitor for anomalies.")
            else:
                try:
                    from sklearn.ensemble import IsolationForest
                    model = IsolationForest(contamination=0.05, random_state=42)
                    preds = model.fit_predict(numeric_df.fillna(0))
                    anomalies = (preds == -1).sum()
                    report.append(f"Detected {anomalies} anomalous rows in numeric columns.")
                except ImportError:
                    report.append("IsolationForest not available for anomaly detection.")

            return "\n".join(report)


# access the data
def load_dataset(path) -> pd.DataFrame:
    """Access and provide the dataset."""
    # Check file extension and load data accordingly
    file_extension = path.split('.')[-1].lower()

    if file_extension == 'csv':
        return pd.read_csv(path)
    elif file_extension == 'parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


# Main execution - Combining all tools into a LangChain Agent
if __name__ == "__main__":
    print("Accessing Data...")

    import argparse
    parser = argparse.ArgumentParser(description="Demo script")
    parser.add_argument("--file", type=str, required=True, help="Dataset file path")
    args = parser.parse_args()

    data_file = args.dataset_path #"output/merged_data.parquet"
    print(f"data_file : {data_file}")
    
    df = load_dataset(data_file)
    data_agent_tools = DataAgentTools(df.head(10))

    # Combine tools into a LangChain agent
    tools = [
        PythonAstREPLTool(locals={"df": df}),
        Tool(name="PlotData",
             func=data_agent_tools.plot_data,
             description="Create plots from the data. Input is Python/Pandas plotting code. "
             "Example: 'validated_df[\"detection_value\"].hist()' or "
             "'validated_df.boxplot(column=\"detection_value\", by=\"project_name\")'"),
        Tool(name="MonitorData",
             func=data_agent_tools.monitor_data,
             description="Monitor data quality and detect anomalies in numeric columns. No input needed."),
    ]

    # Small, CPU-friendly text generation model
    llm = ChatOpenAI(api_key=OPEN_AI_API_KEY, model='gpt-4o-mini')
    df_head = str(df.head(2).to_markdown())

    content = """
    You are a data analysis assistant. working with a pandas dataframe in Python. The name of the dataframe is df. This is the result of print(df.head()):{df_head}.
    You have access to the following tools:
    - search_data(query): Search the dataframe using a pandas query. Return JSON rows.
    - monitor_data(): Detect anomalies in the dataset.
    - plot_data(column): Plot a histogram of a column. Only pass the column name as 'column'.

    Instructions:
    - For plotting, DO NOT send Python code. Always call plot_data(column="column_name").
    - For searching, use pandas query syntax, e.g., 'detection_value > 0.5 & cancer_detected_bool == True'.
    - For monitoring, just call monitor_data() without arguments.
    - Always respond using function calls, do not write raw Python code.
    """.format(df_head=df_head)

    prompt = ChatPromptTemplate(
        input_variables=["agent_scratchpad","input"],
        messages=[
        SystemMessage(content=content) ,
        ("human","{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        stream_runnable=False,
        handle_parsing_errors=True
    )

    # Run Examples

    # Example 1: Search
    ex1 = "I want all samples with detection value  between 0.2 and 1 and cancer_detected is True"
    response = agent_executor.invoke(input={"input": ex1}).get("output", None)
    print(f"Search Q1: {ex1}\nA1: {response}\n")
    # validation to check the agent reliability
    validate_df = df.copy()
    validate_df["cancer_detected_bool"] = validate_df["cancer_detected"] == "Yes"

    # Filter rows
    filtered = validate_df.query("0.2 < detection_value < 1 and cancer_detected_bool == True")
    print("Total matching rows found:", len(filtered))
    print(filtered.head(3).to_dict(orient="records"))

    ex2 = "the last 5 samples according to date_of_run"
    response2 = agent_executor.invoke(input={"input": ex2})
    print(f"Q2: {ex2} \n A2: {response2.get("output")}\n")


