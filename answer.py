import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ratings.dat 파일 경로
file_path = './ml-1m/ratings.dat'

# ratings.dat 파일을 읽어 데이터프레임으로 변환
column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df = pd.read_csv(file_path, sep='::', header=None, names=column_names, engine='python')

# 사용자-아이템 매트릭스로 변환
user_item_matrix = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')

# NaN 값을 0으로 채우기
user_item_matrix = user_item_matrix.fillna(0)

kmeans = KMeans(n_clusters=3, random_state=0)

# 모델 훈련
kmeans.fit(user_item_matrix)

# 클러스터링 결과 라벨
labels = kmeans.labels_

# 결과 출력
user_item_matrix['Cluster'] = labels


cluster_id = 0

### 1번 AU, AVG
print("\n1번 AU, AVG :")

# AU
def recommend_additive_utilitarian(cluster_id, top_n=10):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    item_scores = cluster_data.sum(axis=0).sort_values(ascending=False)
    return item_scores.head(top_n)

# AVG
def recommend_average(cluster_id, top_n=10):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    cluster_data = cluster_data.replace(0.0, np.nan)

    item_scores = cluster_data.mean(axis=0, skipna=True).round(2).sort_values(ascending=False)
    return item_scores.head(top_n)

for cluster_id in range(3):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    rec_AU = recommend_additive_utilitarian(cluster_id)
    rec_AVG = recommend_average(cluster_id)

    print(f"cluster_{cluster_id}_AU: ", rec_AU, "\n")
    print(f"cluster_{cluster_id}_AVG: ", rec_AVG, "\n")


print(f"################\n")

### 2번 CS, AV

print("\n2번 CS, AV :")

# CS
def recommend_simple_count(cluster_id, top_n=10):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    simple_count = (cluster_data != 0.0).sum(axis=0).sort_values(ascending=False)
    return simple_count.head(top_n)

# AV
def recommend_approval_voting(cluster_id, top_n=10):
    threshold = 4
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    approval_voting = (cluster_data >= threshold).sum(axis=0).sort_values(ascending=False)
    return approval_voting.head(top_n)

for cluster_id in range(3):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    rec_SC = recommend_simple_count(cluster_id)
    rec_AV = recommend_approval_voting(cluster_id)

    print(f"cluster_{cluster_id}_SC: ", rec_AU, "\n")
    print(f"cluster_{cluster_id}_AV: ", rec_AVG, "\n")

### 3번 BC

print("\n3번 BC :")

def recommend_borda_count(cluster_id, top_n=10):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    bc = cluster_data.replace(0.0, np.nan)

    # Function to calculate Borda Count scores for each user
    def calculate_borda_count(df):
        bc_scores = pd.DataFrame(index=df.index, columns=df.columns, data=np.nan)
        for user in df.index:
            # Get ratings for the user and drop NaN values
            ratings = df.loc[user].dropna()
            # Rank the items, highest rating gets highest rank
            ranks = ratings.rank(method='average', ascending=False)
            max_rank = len(ranks)
            scores = max_rank - ranks
            bc_scores.loc[user, scores.index] = scores

        # Calculate the final Borda Count by summing scores across users
        bc_sum = bc_scores.sum(axis=0)
        # print(bc_scores)
        return bc_scores, bc_sum

    # Calculate Borda Count
    bc_scores, bc_sum = calculate_borda_count(bc)

    # Display the top n recommended items for the cluster
    return bc_scores, bc_sum.sort_values(ascending=False).head(top_n)


for cluster_id in range(3):
    bc_scores, rec_BC = recommend_borda_count(cluster_id)
    # print(f"cluster_{cluster_id}_bc_scores: ", bc_scores, "\n")
    print(f"cluster_{cluster_id}_BC: ", rec_BC, "\n")


# ### 4번 CR
print("CR 시작:")
print("매우 오래 걸림")
# Function to apply Copeland Rule
def copeland_rule(ratings):
    items = ratings.columns
    n_items = len(items)
    
    # Initialize comparison matrix and Copeland scores
    comparison_matrix = pd.DataFrame(np.zeros((n_items, n_items)), index=items, columns=items)
    copeland_scores = pd.Series(np.zeros(n_items), index=items)
    
    for i in items:
        # print(i)
        for j in items:
            if i != j:
                # Get non-NaN ratings for both items
                valid_ratings = ratings[[i, j]]
                # print(valid_ratings)
                # break
                # Count wins for item i against item j
                wins = (valid_ratings[i] > valid_ratings[j]).sum()
                losses = (valid_ratings[i] < valid_ratings[j]).sum()
                
                if wins > losses:
                    comparison_matrix.loc[i, j] = -1
                elif wins < losses:
                    comparison_matrix.loc[i, j] = 1
                else:
                    comparison_matrix.loc[i, j] = 0
    
    # Calculate Copeland scores
    copeland_scores = comparison_matrix.sum(axis=0)
    
    return comparison_matrix, copeland_scores

# Apply the Copeland Rule
comparison_matrix, copeland_scores = copeland_rule(user_item_matrix)

def recommend_copeland_rule(cluster_id, top_n=10):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    comparison_matrix, copeland_scores = copeland_rule(cluster_data)
    return copeland_scores.sort_values(ascending=False).head(top_n)

for cluster_id in range(3):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    rec_CR = recommend_copeland_rule(cluster_id)

    print(f"cluster_{cluster_id}_CR: ", rec_CR, "\n")
