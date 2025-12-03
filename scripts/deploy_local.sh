#!/bin/bash
set -e

echo "🚀 Deploying Energy MLOps to Local Kubernetes..."

ACR_NAME="arcenergy123"
NAMESPACE="energy-mlops"

# Prerequisites check
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl not installed"; exit 1; }
command -v az >/dev/null 2>&1 || { echo "❌ Azure CLI not installed"; exit 1; }

kubectl cluster-info >/dev/null 2>&1 || { 
  echo "❌ Kubernetes not running. Enable in Docker Desktop."
  exit 1
}

echo "✅ Prerequisites OK"

# Get ACR credentials
echo "🔐 Getting ACR credentials..."
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

# Update manifests
sed -i.bak "s|acrenergyprediction|$ACR_NAME|g" k8s/deployment.yaml
sed -i.bak "s|acrenergyprediction|$ACR_NAME|g" k8s/web-deployment.yaml

# Deploy
kubectl apply -f k8s/namespace.yaml

kubectl create secret docker-registry acr-secret \
  --namespace $NAMESPACE \
  --docker-server=${ACR_NAME}.azurecr.io \
  --docker-username=$ACR_USERNAME \
  --docker-password=$ACR_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/

# Wait for pods
echo "⏳ Waiting for pods..."
kubectl wait --for=condition=ready pod -l app=energy-prediction-api -n $NAMESPACE --timeout=180s || true
kubectl wait --for=condition=ready pod -l app=energy-web -n $NAMESPACE --timeout=180s || true

# Show status
echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
kubectl get pods -n $NAMESPACE
kubectl get svc -n $NAMESPACE
echo ""
echo "🔗 Access:"
echo "  API:  http://localhost:30080/docs"
echo "  Web:  http://localhost:30081"

rm -f k8s/*.bak