import json
import gzip
import glob
import random

# ch
episode_path = "/srv/flash1/kyadav32/datasets/habitat_datasets/datasets/hssd_11100_samples_37_scenes_30objmax/combined_episodes.json.gz"

episode_datas = []
for file in glob.glob(episode_path):
    with gzip.open(file, 'rb') as f:
        episode_datas.append(json.load(f)['episodes'])


episodes_per_scene = {}
# sample 3 episodes per scene
for episode_data in episode_datas:
    for episode in episode_data:
        scene_id = episode['scene_id']
        if scene_id not in episodes_per_scene:
            episodes_per_scene[scene_id] = []
        episodes_per_scene[scene_id].append(episode)

# sample 3 episodes per scene
sampled_episodes = []
for scene_id, episodes in episodes_per_scene.items():
    sampled_episodes.extend(random.sample(episodes, 3))
    print(f"Scene {scene_id}: {len(sampled_episodes)}/{len(episodes)}")

# save the sampled episodes as a json gz file
with gzip.open(f"/srv/flash1/kyadav32/datasets/habitat_datasets/datasets/hssd_11100_samples_37_scenes_30objmax/sampled111episodes.json.gz", 'wb') as f:
    json_str = json.dumps({"episodes": sampled_episodes})
    f.write(json_str.encode('utf-8'))
