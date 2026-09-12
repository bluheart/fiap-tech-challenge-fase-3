from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    'warmup_dag',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['system', 'warmup'],
) as dag:
    for i in range(8): # Use the same number as your parallelism
        BashOperator(
            task_id=f'warmup_{i:02d}',
            bash_command="echo 'spooled'",
        )