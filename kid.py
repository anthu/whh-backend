def send_kid_cloud(path =False):
    if not path:
        dataset_path = "../data/etl/2018_07_31_10_52/"
        all_kid_paths = glob.glob(os.path.join(dataset_path, "*"))
        path = random.choice(all_kid_paths)
    measurement_path = glob.glob(os.path.join(path, "*"))[0]
    with open(measurement_path + '/target.txt') as file:
        target = float(file.read())
    clouds = []
    for file_path in glob.glob(os.path.join(measurement_path+'/pcd', "*")):
        clouds.append(pcl.load(file_path))
    return clouds, target