import os

import utils
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

test_fp_name = "01_14_a0114120050000100033z"
cluster_num = 7

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    all_fp = utils.read_json('data/all_fp.json')
    fp_name_list = []
    for i in range(len(all_fp)):
        name = all_fp[i]["name"]
        fp_name_list.append(name)
    '''
    test_fp_json = utils.find(test_fp_name, all_fp)

    fp_name_list = []
    nr_sim_list = []
    fs_sim_list = []
    rl_sim_list = []
    rc_sim_list = []

    data = []

    for i in range(len(all_fp)):
        name = all_fp[i]["name"]
        nr_sim = utils.number_similarity(test_fp_json["number"], all_fp[i]["number"])
        fs_sim = utils.overallshape_similarity(test_fp_json["overallshape"], all_fp[i]["overallshape"])
        data.append([nr_sim, fs_sim])
        fp_name_list.append(name)
        nr_sim_list.append(nr_sim)
        fs_sim_list.append(fs_sim)

        # rl_sim_list.append(utils.location_similarity(test_fp_json["location"], all_fp[i]["location"]))
        # rc_sim_list.append(utils.connectivity_similarity(test_fp_json["connectivity"], all_fp[i]["connectivity"]))

    sim_dataframe = pd.DataFrame(
        {
            "nr_sim": nr_sim_list,
            "fs_sim": fs_sim_list
        }
    )
    '''

    my_data = pd.read_csv("sim_result.csv")
    # my_data = my_data.drop(columns="nr_sim")
    print(my_data)
    X = np.array(my_data)
    bandwidth = estimate_bandwidth(X, quantile=0.05, n_samples=500)
    meanshift = MeanShift(bandwidth=bandwidth)
    meanshift.fit(X)
    labels = meanshift.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print('Estimated number of clusters: ' + str(n_clusters_))
    y_pred = meanshift.predict(X)
    print(y_pred)
    print(labels)
    print(len(y_pred))
    cv2.waitKey(0)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap="viridis")
    plt.show()

    # for i in range(len(y_pred)):
    #     print(y_pred[i])
    #     fp_img = cv2.imread("data/results/" + fp_name_list[i] + ".png")
    #     if not os.path.exists("classification_result/" + str(y_pred[i])):
    #         os.mkdir("classification_result/" + str(y_pred[i]))
    #     cv2.imwrite("classification_result/" + str(y_pred[i]) + "/" + fp_name_list[i] + ".png", fp_img)

    # pca = PCA(2)
    # df = pca.fit_transform(sim_dataframe)
    # kmeans = cluster.KMeans(n_clusters=cluster_num)
    # label = kmeans.fit_predict(df)
    # print(label)
    #
    # plt.scatter(df[:, 0], df[:, 1])
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.show()
    #
    # # Getting the Centroids
    # centroids = kmeans.cluster_centers_
    # u_labels = np.unique(label)
    #
    # # plotting the results:
    # for i in u_labels:
    #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    # plt.legend()
    # plt.show()


# BDE Main
def conduct_BIG(user_action,clusters_in_view,k,re,prior,n_count,recluster_count,previous_x,is_recluster):
    is_recluster=False
    # 가장 높은 theta가 clusters_in_view에 있는지 확인
    if count_detect(user_action, clusters_in_view, prior, previous_x)==True: # count_detect는 utils.py에 있는 함수
        n_count+=1
    else:
        n_count=n_count
    if n_count==k:
        recluster_count+=1
        n_count=0
        prior=np.ones((len(prior),),dtype=float)/len(prior)
    else:
        prior=posterior(previous_x,user_action,prior) # posterior는 utils.py에 있는 함수
    # next view 생성
    previous_x=Smooth_ViewSearch(user_action,clusters_in_view,prior) # Smooth_ViewSearch는 utils.py에 있는 함수
    # re-clustering 시점 확인
    if recluster_count==re:
        is_recluster=True
        recluster_count=0
    else:
        pass
    return prior, previous_x, is_recluster, n_count, recluster_count

prior, previous_x, is_recluster, n_count, recluster_count = conduct_BIG(user_action,clusters_in_view,k,re,prior,n_count,recluster_count,previous_x,is_recluster)






