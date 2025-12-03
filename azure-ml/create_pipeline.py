"""
Azure ML Pipeline Creator
Submits training pipeline to Azure ML Studio from GitHub Actions
"""

import os
import sys
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# Configuration from environment variables
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


def create_environments(ml_client):
    """Create conda environments for each component"""
    
    # Data prep environment
    dataprep_env = Environment(
        name="dataprep-env",
        description="Environment for data preprocessing",
        conda_file="dataprep/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    dataprep_env = ml_client.environments.create_or_update(dataprep_env)
    print(f"✅ Dataprep environment: {dataprep_env.name}:{dataprep_env.version}")
    
    # Training environment
    training_env = Environment(
        name="training-env",
        description="Environment for model training",
        conda_file="training/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    training_env = ml_client.environments.create_or_update(training_env)
    print(f"✅ Training environment: {training_env.name}:{training_env.version}")
    
    return dataprep_env, training_env


def create_pipeline(ml_client, dataprep_env, training_env):
    """Create the training pipeline"""
    
    # Get dataset
    dataset = ml_client.data.get(name=DATASET_NAME, version=DATASET_VERSION)
    print(f"✅ Dataset: {dataset.name} v{dataset.version}")
    
    # Component 1: Data Preprocessing
    dataprep_component = command(
        name="dataprep",
        display_name="Data Preprocessing",
        code="./dataprep/code",
        command="python dataprep.py --data ${{inputs.raw_data}} --output ${{outputs.processed_data}}",
        environment=f"{dataprep_env.name}:{dataprep_env.version}",
        inputs={
            "raw_data": Input(type="uri_file")
        },
        outputs={
            "processed_data": Output(type="uri_folder")
        }
    )
    
    # Component 2: Train/Test Split
    split_component = command(
        name="split",
        display_name="Train/Test Split",
        code="./components/code",
        command="python traintestsplit.py --data ${{inputs.processed_data}}/preprocessed.csv --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} --test_size 0.2",
        environment=f"{dataprep_env.name}:{dataprep_env.version}",
        inputs={
            "processed_data": Input(type="uri_folder")
        },
        outputs={
            "train_data": Output(type="uri_folder"),
            "test_data": Output(type="uri_folder")
        }
    )
    
    # Component 3: Model Training
    training_component = command(
        name="training",
        display_name="XGBoost Training",
        code="./training/code",
        command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --model_output ${{outputs.model_output}} --n_estimators 200 --max_depth 6 --learning_rate 0.1",
        environment=f"{training_env.name}:{training_env.version}",
        inputs={
            "train_data": Input(type="uri_folder"),
            "test_data": Input(type="uri_folder")
        },
        outputs={
            "model_output": Output(type="uri_folder")
        }
    )
    
    # Define pipeline
    @pipeline(
        name="energy_training_pipeline",
        description="Energy consumption prediction training pipeline",
        compute=COMPUTE_NAME
    )
    def energy_pipeline(raw_data):
        # Step 1: Preprocess
        prep_step = dataprep_component(raw_data=raw_data)
        
        # Step 2: Split
        split_step = split_component(processed_data=prep_step.outputs.processed_data)
        
        # Step 3: Train
        train_step = training_component(
            train_data=split_step.outputs.train_data,
            test_data=split_step.outputs.test_data
        )
        
        return {
            "model_output": train_step.outputs.model_output
        }
    
    # Create pipeline instance
    pipeline_job = energy_pipeline(
        raw_data=Input(type="uri_file", path=dataset.path)
    )
    
    return pipeline_job


def submit_pipeline(ml_client, pipeline_job):
    """Submit pipeline to Azure ML"""
    
    pipeline_job.settings.default_compute = COMPUTE_NAME
    pipeline_job.experiment_name = "energy-consumption-training"
    
    print(f"Submitting pipeline to compute: {COMPUTE_NAME}")
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    
    print(f"\n✅ Pipeline submitted!")
    print(f"Job name: {submitted_job.name}")
    print(f"Job URL: {submitted_job.studio_url}")
    
    # Save job name for later steps
    with open('job_name.txt', 'w') as f:
        f.write(submitted_job.name)
    
    return submitted_job


def main():
    """Main execution"""
    try:
        # Create ML client
        ml_client = create_ml_client()
        
        # Create environments
        dataprep_env, training_env = create_environments(ml_client)
        
        # Create pipeline
        pipeline_job = create_pipeline(ml_client, dataprep_env, training_env)
        
        # Submit pipeline
        submitted_job = submit_pipeline(ml_client, pipeline_job)
        
        print("\n" + "="*50)
        print("✅ SUCCESS - Pipeline submitted to Azure ML")
        print("="*50)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()