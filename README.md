# Installation

- `mamba create -n hab_mt -y python=3.9`
- `mamba install -y habitat-sim==0.3.0 withbullet  headless -c conda-forge -c aihabitat`
- In another directory:
    - `git clone -b v0.3.0 https://github.com/facebookresearch/habitat-lab.git`
    - `cd habitat-lab`
    - `pip install -e habitat-lab`
    - `pip install -e habitat-baselines`
- In another directory:
    - `git clone git@github.com:facebookresearch/eai-vc.git`
    - `cd eai-vc`
    - `git submodule update --init --recursive`
    - `pip install -e ./vc_models`
- `cd` back to project directory
- `pip install -r requirements.txt`
- `pip install -e .`
- Download datasets: 
    - HSSD `python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes`

# Running
bash run_hssd.sh (FCT baseline)
bash run_hssd_ac.sh (Memo - method that makes summaries with accumulation)
bash run_hssd_rmt.sh (Recurrent version of Memo - makes summary but no acccumulation, so memory is effectively overwritten)

# the commands to launch in evaluation mode are in commented out lines