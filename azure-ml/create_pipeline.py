"""
Azure ML Job Submitter - Simplified Single Script
Submits training job to Azure ML from GitHub Actions
"""

import os
import sys
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Configuration
SUBSCRIPTION_ID = os.environ.get('AZURE_SUBSCRIPTION_ID')
RESOURCE_GROUP = os.environ.get('AZURE_RESOURCE_GROUP')
WORKSPACE_NAME = os.environ.get('AZURE_WORKSPACE_NAME')
COMPUTE_NAME = os.environ.get('AZURE_COMPUTE_NAME', 'cpu-cluster')
DATASET_NAME = os.environ.get('DATASET_NAME', 'household-power-consumption')
DATASET_VERSION = os.environ.get('DATASET_VERSION', '1')

def create_ml_client():
    """Create Azure ML client"""
    print(f"Connecting to workspace: {WORKSPACE_NAME}")
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    print("✅ Connected to Azure ML")
    return ml_client


def create_environment(ml_client):
    """Create training environment"""
    env = Environment(
        name="energy-training-env",
        description="Environment for energy consumption model training",
        conda_file="../src/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    env = ml_client.environments.create_or_update(env)
    print(f"✅ Environment: {env.name}:{env.version}")
    return env


def submit_training_job(ml_client, environment):
    """Create and submit training job"""
    
    # Get dataset
    dataset = ml_client.data.get(name=DATASET_NAME, version=DATASET_VERSION)
    print(f"✅ Dataset: {dataset.name} v{dataset.version}")
    
    # Create job
    job = command(
        code="../src",
        command="python train.py --data_path ${{inputs.data_path}} --model_output ${{outputs.model_output}} --n_estimators 200 --max_depth 6 --learning_rate 0.1",
        inputs={
            "data_path": Input(type="uri_file", path=dataset.path)
        },
        outputs={
            "model_output": Output(type="uri_folder")
        },
        environment=f"{environment.name}:{environment.version}",
        compute=COMPUTE_NAME,
        experiment_name="energy-consumption-training",
        display_name="Energy XGBoost Training"
    )
    
    # Submit
    print(f"Submitting job to compute: {COMPUTE_NAME}")
    submitted_job = ml_client.jobs.create_or_update(job)
    
    print(f"\n✅ Job submitted!")
    print(f"Job name: {submitted_job.name}")
    print(f"Job URL: {submitted_job.studio_url}")
    
    # Save job name
    with open('job_name.txt', 'w') as f:
        f.write(submitted_job.name)
    
    return submitted_job


def main():
    """Main execution"""
    try:
        ml_client = create_ml_client()
        environment = create_environment(ml_client)
        submitted_job = submit_training_job(ml_client, environment)
        
        print("\n" + "="*50)
        print("✅ SUCCESS - Job submitted to Azure ML")
        print("="*50)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()