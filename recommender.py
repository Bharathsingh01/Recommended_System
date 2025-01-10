import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(item_name, train_data, top_n=10):
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' Not Found!.")
        return pd.DataFrame()
    tf_vectorizer = TfidfVectorizer(stop_words='english')
    tf_matrix = tf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarity_content = cosine_similarity(tf_matrix, tf_matrix)

    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarity_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_items = similar_items[1:top_n + 1]
    recommended_items_index = [i[0] for i in top_items]
    recommended_items = train_data.iloc[recommended_items_index]['Name']
    return recommended_items

def collaborative_recommendation(target_user_id, train_data, top_n=15):
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_user_indices = user_similarities.argsort()[::-1][1:]
    recommend_items = []
    for user_index in similar_user_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_items = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommend_items.extend(user_item_matrix.columns[not_rated_items][:10])
    recommend_items = train_data[train_data['ProdID'].isin(recommend_items)][['Name']]
    return recommend_items

def recommend_system(user_id, item_name, train_data, top_n=15):
    content = content_based_recommendation(item_name, train_data)
    collaborative = collaborative_recommendation(user_id, train_data)
    recommendations = pd.concat([content, collaborative])
    return recommendations.head(top_n)
