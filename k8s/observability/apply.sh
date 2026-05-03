#!/usr/bin/env bash
# Deploy the observability stack (Prometheus + Loki + Promtail + Grafana)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_DIR="$(cd "$SCRIPT_DIR/../../../llmmllab-gateway" && pwd)"

echo "Deploying observability stack to llmmllab namespace..."

kubectl apply -f "$SCRIPT_DIR/prometheus.yaml"
echo "Prometheus deployed"

kubectl apply -f "$SCRIPT_DIR/alerts.yaml"
echo "Alert rules deployed"

kubectl apply -f "$SCRIPT_DIR/loki.yaml"
echo "Loki deployed"

kubectl apply -f "$SCRIPT_DIR/promtail.yaml"
echo "Promtail deployed"

kubectl apply -f "$SCRIPT_DIR/grafana.yaml"
echo "Grafana deployed"

# Apply gateway routes
echo ""
echo "Applying gateway routes..."
kubectl apply -f "$GATEWAY_DIR/gateway.yaml"
kubectl apply -f "$GATEWAY_DIR/routes.yaml"
echo "Gateway routes deployed"

echo ""
echo "Observability stack deployed successfully!"
echo ""
echo "Access via llmmllab-gateway (192.168.0.71):"
echo "  Grafana:     http://192.168.0.71:3300 (admin / llmmllab-admin)"
echo "  Prometheus:  http://192.168.0.71:3090"
echo "  Loki:        http://192.168.0.71:3100"
