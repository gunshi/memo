import os
import glob
import json
folder = "/srv/flash1/kyadav32/code/gunshi/memory_maze/habitat-transformers/data/scene_datasets/hssd-hab/scenes-uncluttered"

no_scene_filter_files = []
scene_filter_files = []
kind_of_keys = []
for file in glob.glob(os.path.join(folder, "*.scene_instance.json")):
    scene_id = file.split("/")[-1].split(".")[0]

    # check if the file contains scene_filter_file
    with open(file, "r") as f:
        data = json.load(f)
    if "user_defined" not in data or "scene_filter_file" not in data["user_defined"] or "scene_filter" in data["user_defined"]["scene_filter_file"]:
        no_scene_filter_files.append(file)
        kind_of_keys.append(set(data.keys()))
        # add the scene_filter_file to the data
        data["user_defined"] = data.get("user_defined", {})
        data["user_defined"]["scene_filter_file"] = f"scene_filter_files/{scene_id}.rec_filter.json"
        with open(file, "w") as f:
            json.dump(data, f, indent=4)
    else:
        scene_filter_files.append(file)
        kind_of_keys.append(set(data.keys()))
        print(data["user_defined"])

# print(no_scene_filter_files)
print(len(no_scene_filter_files), no_scene_filter_files[0])
print(len(scene_filter_files))

seen_keys = set()
for k in kind_of_keys:
    str_k = ", ".join(k)
    if str_k not in seen_keys:
        print(str_k)
    seen_keys.add(str_k)
