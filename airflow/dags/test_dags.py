from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import pendulum
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Path to the virtual environment
venv_path = os.path.join(project_root, '.venv')
python_path = os.path.join(venv_path, 'bin/python')

# Path to the data_etl.py file
etl_file_path = os.path.join(project_root, 'src', 'data_etl.py')

# timezone
timezone = pendulum.timezone("Asia/Bangkok")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2024, 1, 1, tz=timezone),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id="test_dag",
    default_args=default_args,
    schedule="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
) as dag:
    @task.external_python(task_id='print_project_root', python=python_path)
    def print_project_root(project_root_path):
        import os
        import sys
        
        # Add project root to system path
        sys.path.append(project_root_path)
        
        # Add src directory to system path
        src_path = os.path.join(project_root_path, 'src')
        sys.path.append(src_path)

        # import
        from data_etl import analyse_customer_message_pipeline, PROJECT_DIRECTORY, load_table, get_project_item_path

        # run
        # analyse_customer_message_pipeline()
        df = load_table(get_project_item_path("data/message_table.csv"), list_cols=['user', 'purpose'], datetime_cols=['inserted_at'])
        print(df.shape)

    # Pass project_root to the data_etl task
    print_project_root(project_root)