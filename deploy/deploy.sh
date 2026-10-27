#!/usr/bin/env bash
set -euo pipefail

# Deployment script for EA Skate Native MCP App on Cloud Run
# Follows Dave Wang's Gemini Enterprise architectural security guidelines.

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
SERVICE_NAME="ea-skate-mcp-app"

echo "=========================================================="
echo "Deploying EA Skate MCP App to Cloud Run"
echo "Project: $PROJECT_ID | Region: $REGION"
echo "=========================================================="

# 1. Build and deploy container to Cloud Run
# Note: --no-allow-unauthenticated ensures the app is NEVER public.
#       --no-cpu-throttling ensures background processing won't stall.
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --no-cpu-throttling \
  --min-instances=1 \
  --memory=2Gi \
  --timeout=300

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "Service deployed successfully at: $SERVICE_URL"

# 2. Grant Discovery Engine P4SA IAM access (Roles/run.invoker)
P4SA="service-${PROJECT_NUMBER}@gcp-sa-discoveryengine.iam.gserviceaccount.com"
echo "Granting roles/run.invoker to Discovery Engine P4SA: $P4SA"

gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member="serviceAccount:$P4SA" \
  --role="roles/run.invoker"

echo "=========================================================="
echo "✅ Cloud Run deployment and IAM lockdown complete!"
echo ""
echo "HOW TO REGISTER IN GEMINI ENTERPRISE:"
echo "1. Go to Google Cloud Console -> Vertex AI Search / Gemini Enterprise"
echo "2. Navigate to 'Data Stores' -> 'Data Connectors'"
echo "3. Click 'Add Connector' -> select 'Custom MCP Server'"
echo "4. Enter Endpoint: $SERVICE_URL/mcp"
echo "5. Authentication: Check 'Send X-Serverless-Authorization'"
echo "6. Attach the Data Connector to your Default Assistant"
echo "=========================================================="
