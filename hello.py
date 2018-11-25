from flask import Flask
from flask import request
from flask_cors import CORS
import os
import glob2 as glob
import pcl
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/kid')
def possible_ids():
    path = "../data/etl/2018_07_31_10_52/"
#     path = "/datathon/data/test"
    dirs = os.listdir(path)
    return json.dumps(dirs)

@app.route('/kid/<kid_id>')
def number_of_pointclouds(kid_id):
#     root_path ="/datathon/data/test/"
    root_path = "../data/etl/2018_07_31_10_52/"

    path = root_path +kid_id+"/"
    measurement_path = glob.glob(os.path.join(path, "*"))[0]
    return str(len(os.listdir(measurement_path+'/pcd')))

# @app.route('/kid/<kid_id>/<cloud_id>')
# def get_pointcloud(kid_id,cloud_id, servercall=False):
#     path = "../data/etl/2018_07_31_10_52/"+kid_id+"/"
#     measurement_path = glob.glob(os.path.join(path, "*"))[0]
#     pcd_files = os.listdir(measurement_path+'/pcd')
#     pcd_name = pcd_files[int(cloud_id)]
#     pcd_path = measurement_path + '/pcd/' + pcd_name
#     with open(measurement_path + '/target.txt') as file:
#         target = float(file.read())
#     cloud = pcl.load(pcd_path)
#     points = cloud.to_array()
#     transposed_pts = np.transpose(points)
#     if not servercall:
#         obj = {'target': target, 'points': json.dumps(transposed_pts.tolist())}
#         return json.dumps(obj)
#     else:
#         return points, cloud
    
    
@app.route('/kid/<kid_id>/<cloud_id>')
def get_pointcloud(kid_id,cloud_id):
    
    num_planes = int(request.args.get('num_planes', default=2))
    db_scan = bool(request.args.get('db_scan', default='true')=='true')
    threshold = float(request.args.get('threshold', default=0.05))
    eps = float(request.args.get('db_scan_eps', default=0.1))
    standing = bool(request.args.get('standing', default='true')=='true')
    n_pts = int(request.args.get('n_pts', default=10))
    
    
#     root_path ="/datathon/data/test/"
#     path = root_path +kid_id+"/"
#     measurement_path = glob.glob(os.path.join(path, "*"))[0]
    
    path = "../data/etl/2018_07_31_10_52/"+kid_id+"/"
    measurement_path = glob.glob(os.path.join(path, "*"))[0]
    pcd_files = os.listdir(measurement_path+'/pcd')
    pcd_name = pcd_files[int(cloud_id)]
    pcd_path = measurement_path + '/pcd/' + pcd_name
    with open(measurement_path + '/target.txt') as file:
        target = float(file.read())
#     target = 1
    cloud = pcl.load(pcd_path)
    points = cloud.to_array()
    
    def _find_plane(cloud):
        points = cloud.to_array()
        seg = cloud.make_segmenter()
        seg.set_optimize_coefficients (True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(threshold)
        inliers, normal = seg.segment()
        plane_pts = points[inliers]
        clean_points = np.delete(points, [inliers], 0)
        clean_cloud = pcl.PointCloud(clean_points)
        return clean_cloud, plane_pts, normal
    
    normals = []
    planes = []
    for i in range(num_planes):
        cloud, plane_pts, normal = _find_plane(cloud)
        normals.append(normal)
        planes.append(plane_pts)
        
    points = cloud.to_array()
    if db_scan:
        clustering = DBSCAN(eps=eps).fit(points)
        num_clusters = np.max(clustering.labels_)
        if np.max(clustering.labels_)<= 0:
            kid_pts = points
            outlier_pts = []
        else:
            num_points = [len(points[np.where(clustering.labels_== i)]) for i in range(num_clusters)]
            kid_cluster = num_points.index(np.max(num_points))
            kid_pts = points[np.where(clustering.labels_==kid_cluster)]
            outlier_pts = points[np.where(clustering.labels_!=kid_cluster)]
    else:
        kid_pts = points
        outlier_pts = []

    if standing:
        if num_planes == 1:
            normal = normals[0]
        else:
            normal = normals[1]
        
        #TODO DELETE
#         normal = normals[0]
        
        def _height(point):
            return np.dot(normal[:3],point) + normal[3]

        max_height = 0
        max_idx = 0
        for i, point in enumerate(kid_pts):
            h = _height(point)
            if np.abs(h) > max_height:
                max_height = np.abs(h)
                max_idx = i
        max_pt = [np.asarray([p]) for p in kid_pts[max_idx]]

    else:
        pca = PCA(n_components=1)
        pca.fit(kid_pts)
        transformed_pts = pca.transform(kid_pts)

        #height estimate
        sorted_pts = np.argsort(transformed_pts.flatten())
        top_idx = sorted_pts[-n_pts:]
        low_idx = sorted_pts[:n_pts]

        top_values = [np.asarray(kid_pts[i]) for i in top_idx]
        low_values = [np.asarray(kid_pts[i]) for i in low_idx]
        max_height = np.linalg.norm(np.mean(top_values, axis=0) - np.mean(low_values, axis=0))
        max_pt = top_values + low_values
        max_pt = np.transpose(max_pt)
    
    if num_planes == 2:
        obj = {'target': target,
               'height_estimate' : str(max_height),
               'top_point': json.dumps([p.tolist() for p in max_pt]),
               'kid': json.dumps(np.transpose(kid_pts).tolist()),
               'plane0': json.dumps(np.transpose(planes[0]).tolist()),
               'plane1': json.dumps(np.transpose(planes[1]).tolist()),
               'outliers': json.dumps(np.transpose(outlier_pts).tolist())       
              }
    else:
        obj = {'target': target,
               'height_estimate' : str(max_height),
               'top_point': json.dumps([p.tolist() for p in max_pt]),
               'kid': json.dumps(np.transpose(kid_pts).tolist()),
               'plane0': json.dumps(np.transpose(planes[0]).tolist()),
               'plane1': json.dumps([]),
               'outliers': json.dumps(np.transpose(outlier_pts).tolist())       
              }
    return json.dumps(obj)

    

# @app.route('/kid/<kid_id>/<cloud_id>/wall')
# def get_wall(kid_id,cloud_id, servercall=False):
#     points, cloud = get_pointcloud(kid_id,cloud_id, servercall=True)
#     threshold = float(request.args.get('threshold', default=0.05))
#     seg = cloud.make_segmenter()
#     seg.set_optimize_coefficients (True)
#     seg.set_model_type(pcl.SACMODEL_PLANE)
#     seg.set_method_type(pcl.SAC_RANSAC)
#     seg.set_distance_threshold(threshold)
#     inliers, normal = seg.segment()
#     wall_pts = points[inliers]
#     clean_points = np.delete(cloud.to_array(), [inliers], 0)
#     clean_cloud =  pcl.PointCloud(clean_points)
#     if not servercall:
#         transposed_pts = np.transpose(wall_pts)
#         transposed_clean_pts = np.transpose(clean_points)
#         obj = {'normal': normal, 
#                'points': json.dumps(transposed_clean_pts.tolist()), 
#                'points_2': json.dumps(transposed_pts.tolist())}
#         return json.dumps(obj)
#     else:
#         return cloud, clean_cloud

# @app.route('/kid/<kid_id>/<cloud_id>/floor')
# def get_floor(kid_id,cloud_id, servercall=False):
#     cloud, clean_cloud = get_wall(kid_id,cloud_id, servercall=True)
#     threshold = float(request.args.get('threshold', default=0.05))
#     seg = clean_cloud.make_segmenter()
#     seg.set_optimize_coefficients (True)
#     seg.set_model_type(pcl.SACMODEL_PLANE)
#     seg.set_method_type(pcl.SAC_RANSAC)
#     seg.set_distance_threshold(threshold)
#     inliers, normal = seg.segment()
#     clean_pts = clean_cloud.to_array()
#     floor_pts = clean_pts[inliers]
#     clean_points = np.delete(cloud.to_array(), [inliers], 0)
#     clean_cloud =  pcl.PointCloud(clean_points)
#     if not servercall:
#         transposed_pts = np.transpose(floor_pts)
#         transposed_clean_pts = np.transpose(clean_points)
#         obj = {'normal': normal,
#                'points': json.dumps(transposed_clean_pts.tolist()), 
#                'points_2': json.dumps(transposed_pts.tolist())}
#         return json.dumps(obj)
#     else:
#         return cloud, clean_cloud, normal

# @app.route('/kid/<kid_id>/<cloud_id>/outliers')
# def get_outliers(kid_id,cloud_id, servercall=False):
#     cloud, clean_cloud, normal = get_floor(kid_id,cloud_id, servercall=True)
#     threshold = float(request.args.get('threshold', default=1.0))
#     mean = float(request.args.get('mean', default=50))
#     fil = clean_cloud.make_statistical_outlier_filter()
#     fil.set_mean_k(mean)
#     fil.set_std_dev_mul_thresh(threshold)
#     filtered = fil.filter()
#     filtered_pts = filtered.to_array()
#     if not servercall:
#         transposed_clean_pts = np.transpose(filtered_pts)
#         obj = {'points': json.dumps(transposed_clean_pts.tolist())}
#         return json.dumps(obj)
#     else:
#         return cloud, filtered_pts, normal
    
# @app.route('/kid/<kid_id>/<cloud_id>/height')
# def estimate_height(kid_id,cloud_id):
#     cloud, filtered_pts, normal = get_outliers(kid_id,cloud_id, servercall=True)
#     def _height(point):
#         return np.dot(normal[:3],point) + normal[3]
#     max = 0
#     for i, point in enumerate(filtered_pts):
#         h = _height(point)
#         if h >max: 
#             max = h
#             max_idx = i
#     max_pt = filtered_pts[max_idx]
#     height = max #+ 0.05 # add threshold
#     obj = {'height': height, 'top_point': max_pt.tolist()}
#     return json.dumps(obj)
    
    
if __name__ == '__main__':
    app.run()