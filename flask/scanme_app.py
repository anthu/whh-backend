import numpy as np
import os
import glob2 as glob
import pcl
import json

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# ROOT_PATH = "../data/etl/2018_07_31_10_52/"
ROOT_PATH = '/data/'

@app.route('/kid')
def possible_ids():
    """ returns a list of ids for all kids in /data/etl/2018_07_31_10_52/"""
    dirs = os.listdir(ROOT_PATH)
    return json.dumps(dirs)

@app.route('/kid/<kid_id>')
def number_of_pointclouds(kid_id):
    """ returns the number of available pointclouds in the first measurement with the given kid_id """
    path = ROOT_PATH +kid_id+"/"
    measurement_path = glob.glob(os.path.join(path, "*"))[0]
    return str(len(os.listdir(measurement_path+'/pcd')))


@app.route('/kid/<kid_id>/<cloud_id>')
def get_pointcloud(kid_id,cloud_id):
    """ estimates the height of the kid and returns segmented pointcloud
    
        Args:
            num_planes (int): number of planes to find, can be 1 or 2
            db_scan (bool): indicates if outliers should be detected by clustering with db_scan
            threshold (float): "thickness" of detected planes
            eps (float): distance parameter for db_scan
            standing (bool): indicates if kid is lying down or standing upright
            n_pts (int): number of top/bottom points of the kid that are used for height estimation (only used if standing==False)

        Returns:
            target (float): estimate of the kids height
            top_point (array): points used for height estimation
            kid (array): points identified as "part of the kid"
            plane0 (array): points identified as "wall/ floor"
            plane1 (array): points identified as "wall/ floor"
            outliers (array): points identified as outliers
    """

    # read input params
    num_planes = int(request.args.get('num_planes', default=2))
    db_scan = bool(request.args.get('db_scan', default='true')=='true')
    threshold = float(request.args.get('threshold', default=0.05))
    eps = float(request.args.get('db_scan_eps', default=0.1))
    standing = bool(request.args.get('standing', default='true')=='true')
    n_pts = int(request.args.get('n_pts', default=10))

    # load pcd and target file
    path = ROOT_PATH + kid_id + "/"
    measurement_path = glob.glob(os.path.join(path, "*"))[0]
    pcd_files = os.listdir(measurement_path+'/pcd')
    pcd_name = pcd_files[int(cloud_id)]
    pcd_path = measurement_path + '/pcd/' + pcd_name

    with open(measurement_path + '/target.txt') as file:
        target = float(file.read())
#     target = 1
    cloud = pcl.load(pcd_path)
    points = cloud.to_array()
    
    # fit planes
    def _find_plane(cloud):
        """ fit plane to pointcloud with pcl """
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

    # cluster with db_scan
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

    # estimate heigth
    if standing:
        if num_planes == 1:
            normal = normals[0]
        else:
            normal = normals[1]
        
        #TODO normals need to be switched if floor is recognised before the wall, needs to be made adaptive
#         normal = normals[0]
        
        def _height(point):
            return np.dot(normal[:3],point) + normal[3]

        max_height = 0
        max_idx = 0

        # find highest point
        for i, point in enumerate(kid_pts):
            h = _height(point)
            if np.abs(h) > max_height:
                max_height = np.abs(h)
                max_idx = i
        max_pt = [np.asarray([p]) for p in kid_pts[max_idx]]

    else:
        # find extreme points along first principle component
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
    
    # serialize and return results
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

if __name__ == '__main__':
    app.run(host='0.0.0.0')