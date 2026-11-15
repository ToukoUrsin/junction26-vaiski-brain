# Väiski brain: submission image (policy server on :8000). CPU only; runs the same with or without --gpus.
#   docker build -f aaltonomous/docker/vaiski.Dockerfile -t vaiski:latest .      # from the repo root
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/calibration.py \
     aaltonomous/interface/tasks.py aaltonomous/interface/joint_spec.yaml aaltonomous/interface/calibration.yaml aaltonomous/interface/
COPY aaltonomous/scene/__init__.py aaltonomous/scene/scene.py aaltonomous/scene/scene.yaml aaltonomous/scene/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
COPY aaltonomous/baselines/classical/*.py aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
COPY aaltonomous/vaiski/*.py aaltonomous/vaiski/config.yaml aaltonomous/vaiski/
RUN python3 -c "from aaltonomous.vaiski.policy_server import VaiskiServerPolicy; p=VaiskiServerPolicy(); p.reset('Sort the objects: red into right, blue into left, yellow into center.', 'sort_language', 0); print(p.pol.mapping)"

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.vaiski.policy_server"]
