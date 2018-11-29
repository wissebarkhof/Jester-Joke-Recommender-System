import numpy as np

def recommender_cv(df, ratio = 0.2):
    for idx, user in df.iterrows():
        user = user.iloc[1:]
        filled_out = user.count()
        no_test = int(ratio * filled_out)
        not_na = user.notnull()
        not_na_idx = user.index[not_na].tolist()    
        test_idx = np.random.choice(not_na_idx, no_test)
        train_idx = list(set(not_na_idx) - set(test_idx))
        train, test = user[train_idx], user[test_idx]
        yield train, test

# --- Example
# for train, test in recommender_cv(df.iloc[1:10]):
#     print(len(train), len(test))