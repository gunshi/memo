# HSSD scenes
# make sure git lfs is installed first
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs could not be found. Please install git-lfs first."
    exit 1
fi

if [ -d "data/scene_datasets/hssd-hab" ]; then
    echo "HSSD exists. Skipping HSSD download."
else
    echo "Downloading HSSD scenes..."
    mkdir -p data/scene_datasets/
    pushd data/scene_datasets/

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/hssd/hssd-hab
    cd hssd-hab
    git checkout 34b5f48c343cd7eb65d46e45402c85a004d77c92
    git lfs pull
    git lfs prune -f

    popd
fi

# ExtObjNav
if [ -d "data/datasets/ExtObjNav_HSSD_Diverse" ]; then
    echo "ExtObjNav exists. Skipping ExtObjNav download."
else
    echo "Downloading ExtObjNav..."
    huggingface-cli download aielawady/ExtObjNav --local-dir-use-symlinks False --repo-type dataset --local-dir data/datasets/ExtObjNav_HSSD_Diverse
fi

# VC1 finetuned
if [ -d "data/model_ckpts/cls" ]; then
    echo "VC1 finetuned checkpoint exists. Skipping VC1 finetuned checkpoint download."
else
    echo "Downloading VC1 finetuned checkpoint..."
    huggingface-cli download aielawady/vc1-smallObj --local-dir-use-symlinks False --local-dir data/model_ckpts/cls
fi

# Robot
if [ -d "data/robots/hab_fetch" ]; then
    echo "Robot exists. Skipping Robot download."
else
    echo "Download the Fetch robot model."
    python -m habitat_sim.utils.datasets_download --uids hab_fetch --data-path data/
fi

# Object models
if [ -d "data/objects" ]; then
    echo "Object models exist. Skipping Object models download."
else
    echo "Downloading Object models..."
    python -m habitat_sim.utils.datasets_download --uids ycb --data-path data/
fi