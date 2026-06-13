import trimesh

link5 = trimesh.load("link_5.STL")
right_finger = trimesh.load("right_finger_link.STL")

# Both meshes are in the same global frame originally,
# so the offset between their centroids/origins is the relative transform
offset = right_finger.centroid - link5.centroid  # or use bounding_box.centroid
print(offset)