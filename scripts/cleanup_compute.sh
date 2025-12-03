#!/bin/bash

RESOURCE_GROUP="rg-energy-mlops"
WORKSPACE="ml-energy-prediction"
COMPUTE_NAME="cpu-cluster"

echo "🧹 Cleaning up compute cluster..."

if az ml compute show --name $COMPUTE_NAME \
   --resource-group $RESOURCE_GROUP \
   --workspace-name $WORKSPACE &>/dev/null; then
  
  az ml compute delete \
    --name $COMPUTE_NAME \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE \
    --yes --no-wait
  
  echo "✅ Compute cluster deletion initiated"
else
  echo "ℹ️  Compute cluster doesn't exist"
fi