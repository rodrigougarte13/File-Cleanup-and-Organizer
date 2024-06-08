import os
import shutil
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# extract info about the files
def extract_file_info(folder_path): # returns for each file: [file_name, file_type, file_size, creation_date]
    info_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_type = os.path.splitext(file_path)[1][1:]
            file_size = os.path.getsize(file_path) / 1024
            creation_time = os.path.getctime(file_path)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
            info_list.append([file_name, file_type, file_size, creation_date])
    return info_list


def prepare_features(file_info):
    df = pd.DataFrame(file_info, columns=['file name', 'file type', 'file size (KB)', 'creation date'])
    # featurize the file names
    vectorizer = TfidfVectorizer()
    name_features = vectorizer.fit_transform(df['file name'])
    # one hot encode file types
    type_features = pd.get_dummies(df['file type'])
    # scale features
    scaler = StandardScaler()
    size_features = scaler.fit_transform(df[['file size (KB)']])
    date_features = scaler.fit_transform(pd.to_datetime(df['creation date']).values.reshape(-1, 1))

    # create new df with all the features
    combined_features = pd.concat([
        pd.DataFrame(name_features.toarray()),
        type_features.reset_index(drop=True),
        pd.DataFrame(size_features, columns=['file size']),
        pd.DataFrame(date_features, columns=['creation date'])], axis=1)
    combined_features.columns = combined_features.columns.astype(str)

    return combined_features, df['file name'], vectorizer


def KMeans_clustering(features):
    num_files = features.shape[0]
    min_k = min(5, num_files)
    max_k = max(2, num_files // 4)
    k_values = range(min_k, max_k + 1)
    best_score = -1
    best_k = -1
    rand = np.random.randint(100)

    for k in k_values:
        cluster = KMeans(n_clusters=k, random_state=rand)
        cluster_labels = cluster.fit_predict(features)
        score = silhouette_score(features, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
    print(f'The best k is {best_k} with a silhouette score of {best_score}')

    cluster = KMeans(n_clusters=best_k, random_state=rand)
    cluster_labels = cluster.fit_predict(features)

    return cluster_labels


def generate_folder_names(file_names, labels, vectorizer, n_words):
    folder_names = {}
    for label in set(labels):
        label_indices = np.where(labels == label)[0]
        label_file_names = [file_names[i] for i in label_indices]

        # combine all the numbers in a single string for analysis
        combined_names = " ".join(label_file_names)

        # TF-IDF vectorizer to extract the most important terms
        vectorized = vectorizer.transform([combined_names])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(vectorized.toarray()).flatten()[::-1]
        top_terms = feature_array[tfidf_sorting][:n_words]
        # this could need some work as i generally play with the number to make the most sense out of it
        folder_name = " ".join(top_terms)
        folder_names[label] = folder_name
    return folder_names


def organize_files_clustering(base_path, file_names, labels, folder_names):
    for folder_name in set(folder_names.values()):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for file_name, label in zip(file_names, labels):
        folder_name = folder_names[label]
        source_path = os.path.join(base_path, file_name)
        destination_path = os.path.join(base_path, folder_name, file_name)
        shutil.move(source_path, destination_path)



if __name__ == "__main__":
    folder_path = "C:/Users/rodri/Downloads"
    file_info = extract_file_info(folder_path)
    if file_info:
        features_and_df = prepare_features(file_info)
        features = features_and_df[0]
        df = features_and_df[1]
        vectorizer = features_and_df[2]
        labels = KMeans_clustering((features, df))
        file_names = df['file name']
        folder_names = generate_folder_names(file_names, labels, vectorizer)
        organize_files_clustering(folder_path, file_names, labels, folder_names)
        print("Files organized successfully.")