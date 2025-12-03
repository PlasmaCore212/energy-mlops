"""
Azure ML Pipeline Configuration
Defines and submits the training pipeline to Azure ML
"""

import os
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
from pathlib import Path


def create_ml_client():
    """Create Azure ML client"""
    subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
    resource_group = os.environ.get('AZURE_ML_RESOURCE_GROUP')
    workspace_name = os.environ.get('AZURE_ML_WORKSPACE')
    
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    return ml_client


def get_or_create_compute(ml_client, compute_name="cpu-cluster"):
    """Get or create compute cluster"""
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"Using existing compute: {compute_name}")
    except Exception:
        print(f"Compute {compute_name} not found, assuming it exists...")
        compute = None
    
    return compute_name


def create_environment(ml_client):
    """Create training environment"""
    env_name = "energy-training-env"
    
    # Check if environment exists
    try:
        env = ml_client.environments.get(env_name, version="1")
        print(f"Using existing environment: {env_name}")
    except Exception:
        print(f"Creating new environment: {env_name}")
        
        env = Environment(
            name=env_name,
            description="Environment for energy consumption prediction training",
            conda_file="src/conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        env = ml_client.environments.create_or_update(env)
    
    return env


def create_training_job(ml_client, compute_name, environment):
    """Create and configure training job"""
    
    # Get dataset
    data_asset = ml_client.data.get(name="household-power-consumption", version="1")
    
    # Create training job
    job = command(
        code="./src",
        command="python train.py --data_path ${{inputs.data_path}} --model_output ${{outputs.model_output}} --n_estimators 200 --max_depth 6 --learning_rate 0.1",
        inputs={
            "data_path": Input(type="uri_file", path=data_asset.path)
        },
        outputs={
            "model_output": {"type": "uri_folder"}
        },
        environment=f"{environment.name}:{environment.version}",
        compute=compute_name,
        experiment_name="energy-consumption-training",
        display_name="Energy Consumption XGBoost Training"
    )
    
    return job


def submit_pipeline(ml_client):
    """Submit training pipeline"""
    print("Setting up Azure ML pipeline...")
    
    # Get/create compute
    compute_name = get_or_create_compute(ml_client)
    
    # Create environment
    environment = create_environment(ml_client)
    
    # Create training job
    job = create_training_job(ml_client, compute_name, environment)
    
    # Submit job
    print("Submitting training job...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print(f"✅ Job submitted successfully!")
    print(f"Job name: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")
    
    return returned_job


def register_model(ml_client, job_name):
    """Register the trained model"""
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes
    
    print("Registering model...")
    
    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/model_output",
        name="energy-consumption-model",
        description="XGBoost model for household energy consumption prediction",
        type=AssetTypes.CUSTOM_MODEL
    )
    
    registered_model = ml_client.models.create_or_update(model)
    
    print(f"✅ Model registered: {registered_model.name} (version {registered_model.version})")
    
    return registered_model


def main():
    """Main pipeline execution"""
    
    # Create ML client
    ml_client = create_ml_client()
    
    # Submit pipeline
    job = submit_pipeline(ml_client)
    
    # Wait for job completion (optional for GitHub Actions)
    # ml_client.jobs.stream(job.name)
    
    print(f"\n📊 Monitor your job at: {job.studio_url}")
    print(f"Job ID: {job.name}")


if __name__ == "__main__":
    main()