echo "Create cluster"

kind create cluster --name workshop-end2end

echo "Install argilla"

helm repo add stable https://charts.helm.sh/stable
helm repo add elastic https://helm.elastic.co
helm repo update
helm install elastic-operator elastic/eck-operator

git clone https://github.com/argilla-io/argilla.git
helm dependency build argilla/examples/deployments/k8s/argilla-chart
helm install my-argilla-server argilla/examples/deployments/k8s/argilla-chart
rm -rf argilla




echo "Install langfuse"
helm repo add langfuse https://langfuse.github.io/langfuse-k8s
helm repo update
helm install langfuse langfuse/langfuse

echo "Install signoz"

DEFAULT_STORAGE_CLASS=$(kubectl get storageclass -o=jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')
kubectl patch storageclass "$DEFAULT_STORAGE_CLASS" -p '{"allowVolumeExpansion": true}'

helm repo add signoz https://charts.signoz.io
helm install signoz signoz/signoz