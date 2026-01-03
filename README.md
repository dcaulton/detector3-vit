# detector3
detector project to run with a Frigate network and Nvidia acceleration
(cloned from detection1)

This is an area for looking at vit (vision transformer) detection models.  It should allow me to try a few different versions.
It runs cleanly in the my homelab setup (mini pc, lots of ram and disk, 1080 Ti GPU).  The following features work:
- written in python, all mainline logic is in src/app.py
- connects to MQTT, responds to all frigate topics
- a slice of the GPU is allocated
- has access to a 500GB PVC
- runs detection on all frigate snapshots using the specified vit model
- saves resulting image, logs it to MLFlow
