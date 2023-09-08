from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy


reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('path/to/dataset', reader=reader)


trainset, testset = train_test_split(data, test_size=0.25)


sim_options = {
    'name': 'cosine',
    'user_based': True
}

model = KNNBasic(sim_options=sim_options)


model.fit(trainset)


predictions = model.test(testset)


rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')


user_id = '1'
items_to_recommend = 10


user_ratings = trainset.ur[int(user_id)]
items_rated_by_user = set([item for (item, _) in user_ratings])
items_not_rated_by_user = list(set(trainset.all_items()) - items_rated_by_user)


item_ratings = [(item, model.predict(user_id, item).est) for item in items_not_rated_by_user]


item_ratings.sort(key=lambda x: x[1], reverse=True)


top_recommendations = item_ratings[:items_to_recommend]


for item, rating in top_recommendations:
    print(f'Recommended Item ID: {item}, Predicted Rating: {rating}')

