# MAKE SURE TO RUN THIS SCRIPT AT ROOT OF PROJECT DIRECTORY
# WHERE IS THE DIRECTORY CONTAINING YOUR PYTHON ENVIRONMENTS

#!/bin/zsh
set -e

# define temporary $AIRFLOW_HOME
export AIRFLOW_HOME=$(pwd)/airflow

# Activate the Airflow virtual environment
source .airflow_venv/bin/activate

# Check if the database is already initialized
if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "Initializing Airflow database..."
    airflow db init

    echo "Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
else
    echo "Airflow database already initialized, skipping initialization"
fi

# Start the scheduler at daemon mode
airflow scheduler -D &

# Start the webserver at daemon mode    
airflow webserver -D --port 8080