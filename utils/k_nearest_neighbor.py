import numpy as np

def euclidean_distance(feat1, feat2):
    feat1 = np.array(feat1, dtype=float)
    feat2 = np.array(feat2, dtype=float)
    
    dists = np.zeros((len(feat1), len(feat2)))
    for i in range(len(feat1)):
        for j in range(len(feat2)):
            dists[i, j] = np.sqrt(np.sum(np.power(feat1[i] - feat2[j], 2)))
    
    min_distance = np.min(dists)
    return min_distance

def find_nearest_neighbors(features_list, audio_features_dataset, k=5):
    """Tìm các item giống nhất với features_list trong audio_features_dataset bằng thuật toán KNN."""
    distances = []
    
    for item in audio_features_dataset:
        avg_distances = []
        for feature_name in features_list:
            if feature_name == 'Audio_File':
                continue
            avg_distance = euclidean_distance(features_list[feature_name], item[feature_name])
            avg_distances.append(avg_distance)
        
        avg_distance = np.mean(avg_distances)
        distances.append((item['Audio_File'], avg_distance))
    
    # Sắp xếp các item theo khoảng cách tăng dần
    distances.sort(key=lambda x: x[1])
    
    # Lấy ra k item gần nhất
    nearest_neighbors = [item[0] for item in distances[:k]]
    
    return nearest_neighbors

# Sử dụng hàm để tìm 5 item giống nhất
