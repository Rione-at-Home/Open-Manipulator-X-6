import trimesh
link5 = trimesh.load("link_5.STL")
left = trimesh.load("left_finger.STL")
right = trimesh.load("right_finger.STL")

print("left offset:", left.centroid - link5.centroid)
print("right offset:", right.centroid - link5.centroid)

print("left joint origin - left centroid offset:", 
      (-0.019263, -1.6285, -1.7856) - (left.centroid - link5.centroid))

