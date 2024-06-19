import pandas as pd
import numpy as np

data = {
    'user_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    'item_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'rating': [2, 3, 5, 4, None, 3, 2, 5, 4, 3, 2, None, 1, 3, 4]
}

df = pd.DataFrame(data)
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

user_item_matrix = user_item_matrix.fillna(0)

user_item_matrix.head()

user_item_matrix['Cluster'] = np.array([0,0,0])

# print(user_item_matrix)


print(user_item_matrix)
print()
cluster_id = 0

### 1번 AU, AVG
for cluster_id in range(1):
    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)
    item_sum_scores = cluster_data.sum(axis=0)

    cluster_data = cluster_data.replace(0.0, np.nan)
    item_avg_scores = cluster_data.mean(axis=0, skipna=True).round(2)
    
    # print("cluster data")
    # print(cluster_data)
    # print()
    # Step 3: Combine the results into a new DataFrame
    result_df = pd.DataFrame({
        'AU': item_sum_scores,
        'Avg': item_avg_scores

    }).T

    result_df.to_csv(f'Simple_computation_group{cluster_id}.csv')
print("1번 AU, AVG :")
print(result_df)


### 2번 SC, AV
for cluster_id in range(1):

    cluster_data = user_item_matrix[user_item_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)

    # Calculate Simple Count (SC)
    simple_count = (cluster_data != 0.0).sum(axis=0)

    # Calculate Approval Voting (AV) with threshold = 4
    threshold = 4
    approval_voting = (cluster_data >= threshold).sum(axis=0)

    # Create the final DataFrame
    result_df = pd.DataFrame({
        'SC': simple_count,
        'AV': approval_voting
    }).T

    result_df.to_csv(f'Counting_the_rating{cluster_id}.csv')
print("\n2번 SC, AV:")
print(result_df)

# # Borda Count 계산
# bc_scores = calculate_borda_count(df)
# # 결과 출력
print("\n3번 BC")

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
            ranks = ratings.rank(ascending=False)
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
    return bc_scores, bc_sum.sort_values(ascending=False).head(top_n), bc_sum


bc_scores, rec_BC, bc_sum = recommend_borda_count(0)
print(bc_scores, "\n")
print(bc_sum, "\n")

print(rec_BC, "\n")

################################

# 결과 확인 -> nan 대신 0.0 써야 됨
ratings = pd.DataFrame({
    1: [2.0, 3.0, 2.0],
    2: [3.0, 2.0, 0.0],
    3: [5.0, 5.0, 1.0],
    4: [4.0, 4.0, 3.0],
    5: [0.0, 3.0, 4.0]
}, index=[1, 2, 3])

# Function to apply Copeland Rule
def copeland_rule(ratings):
    items = ratings.columns
    n_items = len(items)
    
    # Initialize comparison matrix and Copeland scores
    comparison_matrix = pd.DataFrame(np.zeros((n_items, n_items)), index=items, columns=items)
    copeland_scores = pd.Series(np.zeros(n_items), index=items)
    
    for i in items:
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
comparison_matrix, copeland_scores = copeland_rule(ratings)

# Print results

print("\n4번 CR")

print("Comparison Matrix:")
print(comparison_matrix)
print("\nCopeland Scores:")
print(copeland_scores)
