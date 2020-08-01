import numpy as np
from sklearn.neighbors import NearestNeighbors
from .herding import herding_selection
from .kmeans import kmeans_sample

def get_overlap_region_exemplars(train_dict):
    all_features = np.vstack(list(train_dict.values()))
    k =  int(8 * np.log10(len(all_features))) # math.ceil(exemp_size / 2)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(all_features)

    length_by_classes = [len(each) for each in list(train_dict.values())]
    list_of_sums = [sum(length_by_classes[:idx + 1]) for idx in range(len(length_by_classes))]
    dict_of_sums = {sum_val: key for sum_val, key in zip(list_of_sums, train_dict.keys())}

    label_to_indices_dict = {key: [] for key in train_dict.keys()}
    for label, features in train_dict.items():
        for idx, data_point in enumerate(features):
            count_of_neighbours = {}
            distances, indices = nbrs.kneighbors(data_point.reshape(1, -1))
            for index in indices[0][1:]: # since first index always contains the element itself
                # get sum value that is closest to this index
                nearest_sum = min(list_of_sums, key = lambda  x: (x-index) if x >= index else max(list_of_sums))
                nearest_class = dict_of_sums[nearest_sum]
                # print(nearest_class, label)
                if nearest_class not in count_of_neighbours:
                    count_of_neighbours[nearest_class] = 1
                else:
                    count_of_neighbours[nearest_class] += 1
            dominant_neighbour = max(count_of_neighbours.items(), key=lambda x: x[1])[0]
            if dominant_neighbour == label:
                n_c = len(count_of_neighbours)
                if n_c > 1:
                    lamda = 0.3 # suggested value [0.1, 0.3]
                    del count_of_neighbours[dominant_neighbour]
                    second_dominant_nbr = max(count_of_neighbours.items(), key=lambda x: x[1])[1] / k
                    # print(second_dominant_nbr*2, 1/ n_c, 1 / n_c + lamda)
                    if second_dominant_nbr >= 1 / n_c and second_dominant_nbr <= 1 / n_c + lamda:
                        label_to_indices_dict[label].append(idx)

    return label_to_indices_dict

def get_edge_region_exemplars(train_dict):
    all_features = np.vstack(list(train_dict.values()))
    k =  int(8 * np.log10(len(all_features)))#math.ceil(exemp_size / 3) # int(5 * np.log10(len(all_features)))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(all_features)

    length_by_classes = [len(each) for each in list(train_dict.values())]
    list_of_sums = [sum(length_by_classes[:idx + 1]) for idx in range(len(length_by_classes))]
    dict_of_sums = {sum_val: key for sum_val, key in zip(list_of_sums, train_dict.keys())}

    label_to_indices_dict = {key: [] for key in train_dict.keys()}
    for label, features in train_dict.items():
        if len(features) > 1:
            k_e = int(3 * np.log(len(features)))  # math.ceil(exemp_size / 3) # int(5 * np.log10(len(all_features)))
            nbrs_l = NearestNeighbors(n_neighbors=k_e, algorithm='auto').fit(features)
            for idx, data_point in enumerate(features):
                count_of_neighbours = {}
                distances, indices = nbrs.kneighbors(data_point.reshape(1, -1))
                for index in indices[0][1:]: # since first index always contains the element itself
                    # get sum value that is closest to this index
                    nearest_sum = min(list_of_sums, key = lambda  x: (x-index) if x >= index else max(list_of_sums))
                    nearest_class = dict_of_sums[nearest_sum]
                    # print(nearest_class, label)
                    if nearest_class not in count_of_neighbours:
                        count_of_neighbours[nearest_class] = 1
                    else:
                        count_of_neighbours[nearest_class] += 1
                dominant_neighbour = max(count_of_neighbours.items(), key=lambda x: x[1])[0]
                if dominant_neighbour == label:
                    lamda, gamma = 0.1, 0.25
                    n_c = len(count_of_neighbours)
                    if n_c > 1:
                        del count_of_neighbours[dominant_neighbour]
                        second_dominant_nbr_score = max(count_of_neighbours.items(), key=lambda x: x[1])[1] / k
                        if second_dominant_nbr_score*2 > (lamda + 1 / n_c):  # eqn. (4)
                            # check for k_e nearest neighbours that:
                            distances_, indices_ = nbrs_l.kneighbors(data_point.reshape(1, -1))
                            normal_vecs, difference_vecs = [], []
                            for index in indices_[0][1:]:
                                difference = features[index] - data_point
                                difference_vecs.append(difference)
                                normal_vecc = difference / np.linalg.norm(difference)
                                normal_vecs.append(normal_vecc)
                            sum_normal_vec = np.sum(np.array(normal_vecs), axis=0)
                            I = 0
                            for differ in difference_vecs:
                                theta = differ @ sum_normal_vec
                                if theta > 0:
                                    I += 1
                            l_i = I / k_e
                            if l_i >= (1 - gamma):
                                label_to_indices_dict[label].append(idx)
    return label_to_indices_dict

def get_interior_region_exemplars(train_dict, dict_of_means, exemp_size_per_class):
    label_to_indices_dict = {}
    for label, size in exemp_size_per_class.items():
        if size > 0:
            mean_of_class = dict_of_means[label]
            # top_k_indices = herding_selection(train_dict[label], size, mean_=mean_of_class)
            top_k_indices = kmeans_sample(train_dict[label], size)
            label_to_indices_dict[label] = top_k_indices
    return label_to_indices_dict

# def get_interior_region_exemplars(train_dict, dict_of_means, exemp_size_per_class):
#     label_to_indices_dict = {}
#     for label, size in exemp_size_per_class.items():
#         if size > 0:
#             mean_of_class = dict_of_means[label]
#             cosine_sims = np.array([np.dot(x, mean_of_class) / (np.linalg.norm(x) * np.linalg.norm(mean_of_class)) for
#                                     x in train_dict[label]])
#             try:
#                 top_k_indices = np.argpartition(cosine_sims, -size)[-size:] # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
#                 label_to_indices_dict[label] = top_k_indices
#             except ValueError:
#                 print(cosine_sims, size, label, len(train_dict[label]))
#     return label_to_indices_dict

# def get_new_exemplars(dict_of_features, dict_of_means, exemp_size):
#     normalised_features_dict = {key: feature / np.linalg.norm(feature) for key, feature in dict_of_features.items()}
#     overlapping_exemplars_indices = get_overlap_region_exemplars(normalised_features_dict, exemp_size)
#     overlapping_exemplars = {label: np.array(features)[overlapping_exemplars_indices[label][:exemp_size]] for label, features in
#                      dict_of_features.items()}
#     filtered_features = {label: np.delete(features, overlapping_exemplars_indices[label][:exemp_size], axis=0) for
#                                  label, features in dict_of_features.items() }
#     normalised_features_dict = {key: feature / np.linalg.norm(feature) for key, feature in filtered_features.items()}
#     edge_exemplar_indices = get_edge_region_exemplars(normalised_features_dict, exemp_size)
#     edge_exemplars = {label: np.array(features)[edge_exemplar_indices[label]] for label, features in
#                      filtered_features.items()}
#     reqd_exemplars_per_class = {key: exemp_size - len(features) for key, features in overlapping_exemplars.items()}
#     # print(reqd_exemplars_per_class)
#     total_exemplars = {label: np.vstack((overlapping_exemplars[label], edge_exemplars[label][:reqd_exemplars_per_class[label]])) for label in
#                        dict_of_features.keys()}
#
#     filtered_features = {label: np.delete(features, edge_exemplar_indices[label][:exemp_size], axis=0) for
#                          label, features in filtered_features.items()}
#     normalised_features_dict = {key: feature / np.linalg.norm(feature) for key, feature in filtered_features.items()}
#     reqd_exemplars_per_class = {key: exemp_size - len(features) for key, features in total_exemplars.items()}
#     # print(reqd_exemplars_per_class, exemp_size)
#     interior_exemplar_indices = get_interior_region_exemplars(normalised_features_dict, dict_of_means, reqd_exemplars_per_class)
#     interior_exemplars = {label: np.array(features)[interior_exemplar_indices[label]] for label, features in filtered_features.items()
#                           if reqd_exemplars_per_class[label] > 0}
#     total_exemplars = {label: np.vstack((interior_exemplars[label], total_exemplars[label])) if
#     reqd_exemplars_per_class[label] > 0 else total_exemplars[label] for label in dict_of_features.keys()}
#     # print({key: len(features) for key, features in total_exemplars.items()})
#     return total_exemplars

def get_new_exemplars(dict_of_features, normalised_features_dict, dict_of_means, exemp_size_dict):
    overlapping_exemplars_indices = get_overlap_region_exemplars(normalised_features_dict)
    overlapping_exemplars = {label: np.array(features)[overlapping_exemplars_indices[label][:exemp_size_dict[label]]] for label, features in
                             dict_of_features.items()}
    filtered_features = {label: np.delete(features, overlapping_exemplars_indices[label][:exemp_size_dict[label]], axis=0) for
                                 label, features in dict_of_features.items()}
    normalised_features_dict = {label: np.delete(features, overlapping_exemplars_indices[label][:exemp_size_dict[label]], axis=0) for
                                 label, features in normalised_features_dict.items()}

    edge_exemplar_indices = get_edge_region_exemplars(normalised_features_dict)
    edge_exemplars = {label: np.array(features)[edge_exemplar_indices[label]] for label, features in
                     filtered_features.items()}
    reqd_exemplars_per_class = {key: exemp_size_dict[key] - len(features) for key, features in overlapping_exemplars.items()}
    # print(reqd_exemplars_per_class)
    total_exemplars = {label: np.vstack((overlapping_exemplars[label], edge_exemplars[label][:reqd_exemplars_per_class[label]])) for label in
                       dict_of_features.keys()}

    filtered_features = {label: np.delete(features, edge_exemplar_indices[label][:exemp_size_dict[label]], axis=0) for
                         label, features in filtered_features.items()}
    normalised_features_dict = {label: np.delete(features, edge_exemplar_indices[label][:exemp_size_dict[label]], axis=0) for
                         label, features in normalised_features_dict.items()}
    reqd_exemplars_per_class = {key: exemp_size_dict[key] - len(features) for key, features in total_exemplars.items()}
    interior_exemplar_indices = get_interior_region_exemplars(normalised_features_dict, dict_of_means, reqd_exemplars_per_class)
    interior_exemplars = {label: np.array(features)[interior_exemplar_indices[label]] for label, features in filtered_features.items()
                          if reqd_exemplars_per_class[label] > 0}
    total_exemplars = {label: np.vstack((interior_exemplars[label], total_exemplars[label])) if
    reqd_exemplars_per_class[label] > 0 else total_exemplars[label] for label in dict_of_features.keys()}
    return total_exemplars