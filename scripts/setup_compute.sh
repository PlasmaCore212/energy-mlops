#!/bin/bash
set -e

RESOURCE_GROUP="rg-energy-mlops"
WORKSPACE="ml-energy-prediction"
COMPUTE_NAME="cpu-cluster"

echo "🔍 Checking if compute cluster exists..."

if az ml compute show --name $COMPUTE_NAME \
   --resource-group $RESOURCE_GROUP \
   --workspace-name $WORKSPACE &>/dev/null; then
  echo "✅ Compute cluster '$COMPUTE_NAME' already exists"
else
  echo "🚀 Creating compute cluster '$COMPUTE_NAME'..."
  
  az ml compute create \
    --name $COMPUTE_NAME \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE \
    --type amlcompute \
    --size Standard_DS3_v2 \
    --min-instances 0 \
    --max-instances 4 \
    --idle-time-before-scale-down 120
  
  echo "✅ Compute cluster created"
fi