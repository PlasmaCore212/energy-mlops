#!/bin/bash
# Energy MLOps Deployment Helper Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Energy MLOps Deployment Helper${NC}"
echo "================================"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

command -v az >/dev/null 2>&1 || { echo -e "${RED}Azure CLI not installed${NC}"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl not installed${NC}"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker not installed${NC}"; exit 1; }

echo -e "${GREEN}✓ All prerequisites installed${NC}"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Function: Get AKS credentials
get_aks_credentials() {
    echo -e "\n${YELLOW}Getting AKS credentials...${NC}"
    az aks get-credentials \
        --name ${AKS_CLUSTER} \
        --resource-group ${AKS_RESOURCE_GROUP} \
        --overwrite-existing
    echo -e "${GREEN}✓ AKS credentials configured${NC}"
}

# Function: Deploy to AKS
deploy_to_aks() {
    echo -e "\n${YELLOW}Deploying to AKS...${NC}"
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Create ACR secret
    kubectl create secret docker-registry acr-secret \
        --namespace energy-mlops \
        --docker-server=${ACR_NAME}.azurecr.io \
        --docker-username=${ACR_USERNAME} \
        --docker-password=${ACR_PASSWORD} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    
    echo -e "${GREEN}✓ Deployed to AKS${NC}"
}

# Function: Check deployment status
check_status() {
    echo -e "\n${YELLOW}Checking deployment status...${NC}"
    kubectl get pods -n energy-mlops
    kubectl get svc -n energy-mlops
}

# Function: Get service URLs
get_urls() {
    echo -e "\n${YELLOW}Getting service URLs...${NC}"
    API_IP=$(kubectl get svc energy-prediction-api -n energy-mlops -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$API_IP" ]; then
        echo -e "${YELLOW}⏳ LoadBalancer IP not yet assigned${NC}"
    else
        echo -e "${GREEN}API URL: http://${API_IP}${NC}"
        echo -e "${GREEN}Swagger Docs: http://${API_IP}/docs${NC}"
    fi
}

# Main menu
echo -e "\n${YELLOW}What would you like to do?${NC}"
echo "1. Get AKS credentials"
echo "2. Deploy to AKS"
echo "3. Check deployment status"
echo "4. Get service URLs"
echo "5. All of the above"
echo "6. Exit"

read -p "Enter choice [1-6]: " choice

case $choice in
    1) get_aks_credentials ;;
    2) deploy_to_aks ;;
    3) check_status ;;
    4) get_urls ;;
    5)
        get_aks_credentials
        deploy_to_aks
        check_status
        get_urls
        ;;
    6) exit 0 ;;
    *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
esac

echo -e "\n${GREEN}Done!${NC}"
