# detector1
detector project to run with a Frigate network and Nvidia acceleration

This is intended to be a template.  It runs cleanly in the my homelab setup (mini pc, lots of ram and disk, 1080 Ti GPU).  The following features work:
- written in python, all mainline logic is in src/app.py
- connects to MQTT, responds to all frigate topics
- a slice of the GPU is allocated
- has access to a 500GB PVC
- does some quick matrix math and opencv work to verify the GPU actually works
- saves resulting image, logs it to MLFlow

To make a new project based on this one:
- Fork the project with a good name, maybe detector2-yolo
- Make a new namespace, eg detection2
- copy the secrets (MQTT creds for now) to the new namespace *
- Update k8s/deployment.yaml - set replicas to 1 to enable the project 
- Update k8s files otherwise, set all detection1 to detection2
- In the Homelab repo, make a new folder in apps, with an appliction.yaml like that for detection1
- Resync Homelab in ArgoCD


heres how to copy the secrets:
kubectl get secret detection-app-secrets -n detection1 -o yaml \
  | sed 's/namespace: detection1/namespace: detection2/' \
  | kubectl apply -f -

