import sys
import math

def distancia_euclidiana(punto1, punto2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(punto1, punto2)))

def linkage_distancia(cluster_a, cluster_b, dist_matrix, linkage):
    distancias = [dist_matrix[i][j] for i in cluster_a for j in cluster_b]
    if linkage == 0:   # Single Linkage (Min)
        return min(distancias)
    elif linkage == 1: # Complete Linkage (Max)
        return max(distancias)
    elif linkage == 2: # Average Linkage (Mean)
        return sum(distancias) / len(distancias)
    else:
        raise ValueError("Linkage must be 0, 1, or 2")

def data_lecture():
    input_data = sys.stdin.read().split()
    if not input_data: return None
    
    N, P, K, L = map(int, input_data[:4])
    coords = []
    idx = 4
    for _ in range(N):
        coords.append([float(x) for x in input_data[idx:idx+P]])
        idx += P
    return N, P, K, L, coords

def process_aglomerative_clustering(N, P, K, L, coords):
    # Pre-calculate distance matrix to avoid redundant math
    dist_matrix = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            dist = distancia_euclidiana(coords[i], coords[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    # Initialization: Each point is a cluster
    clusters = {i: [i] for i in range(N)}
    next_id = N
    history = []
    
    # Looking for the nearest pair
    while len(clusters) > K:
        min_dist = float('inf')
        best_pair = None
        
        current_ids = sorted(clusters.keys())
        
        for i in range(len(current_ids)):
            for j in range(i + 1, len(current_ids)):
                id_a, id_b = current_ids[i], current_ids[j]
                d = linkage_distancia(clusters[id_a], clusters[id_b], dist_matrix, L)
                
                # Selection criteria with ID tie-break
                # Use a small epsilon for floating point precision
                if d < min_dist - 1e-9:
                    min_dist = d
                    best_pair = (id_a, id_b)
                elif abs(d - min_dist) < 1e-9:
                    if best_pair is None or id_a < best_pair[0] or (id_a == best_pair[0] and id_b < best_pair[1]):
                        best_pair = (id_a, id_b)

        id_a, id_b = best_pair
        history.append(f"{id_a} {id_b} {next_id} {min_dist:.4f}")
        
        # Create new cluster and delete the old ones
        clusters[next_id] = clusters[id_a] + clusters[id_b]
        del clusters[id_a]
        del clusters[id_b]
        next_id += 1
    
    # Print merge history
    for line in history:
        print(line)
    
    # Generate final labels
    # Sort clusters by the smallest original point index they contain
    final_clusters_list = sorted(clusters.values(), key=min)
    point_labels = [0] * N
    
    for label, cluster_points in enumerate(final_clusters_list):
        for p_idx in cluster_points:
            point_labels[p_idx] = label
            
    print(*(point_labels))
    
if __name__ == "__main__":
    result = data_lecture()
    if result:
        N, P, K, L, coords = result
        process_aglomerative_clustering(N, P, K, L, coords)