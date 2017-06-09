# coding=utf-8
import pandas as pd
import numpy as np
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
ad = pd.read_csv('../ad.csv')
user = pd.read_csv('../user.csv')

train.drop(["conversionTime"], axis=1, inplace=True)
test.drop(["instanceID"], axis=1, inplace=True)
data = pd.concat([train, test], axis=0)
data["day"] = (data["clickTime"] / 10000).astype(int)
data = pd.merge(data, ad, on="creativeID", how='left')
data.fillna(0, inplace=True)
data = pd.merge(data, user, on="userID", how='left')
data["count"] = np.zeros(data.shape[0])

temp = data[data["day"] >= 17]
temp = temp[temp["day"] < 24]
gb = temp.groupby(["positionID", "connectionType"]).count().rename(
    columns={'clickTime': 'position_connectiontype_count'}).reset_index()
gb = gb.loc[:,
            ["positionID", "connectionType", "position_connectiontype_count"]]
temp = pd.merge(
    data[data["day"] == 24],
    gb,
    on=["positionID", "connectionType"],
    how='left')
result = temp

gb = temp.groupby(["advertiserID", "positionID"]).count().rename(
    columns={'count': 'advertiser_position_count'}).reset_index()
gb = gb.loc[:, ["advertiserID", "positionID", "advertiser_position_count"]]
temp = pd.merge(result, gb, on=["advertiserID", "positionID"], how='left')
reult = temp

gb = temp.groupby(["gender", "positionID"]).count().rename(
    columns={'count': 'gender_position_count'}).reset_index()
gb = gb.loc[:, ["gender", "positionID", "gender_position_count"]]
temp = pd.merge(result, gb, on=["gender", "positionID"], how='left')
reult = temp

for day in xrange(25, 32):
    temp = data[data["day"] >= day - 7]
    temp = temp[temp["day"] < day]
    gb = temp.groupby(["positionID", "connectionType"]).count().rename(
        columns={'clickTime': 'position_connectiontype_count'}).reset_index()
    gb = gb.loc[:, [
        "positionID", "connectionType", "position_connectiontype_count"
    ]]
    temp = pd.merge(
        data[data["day"] == day],
        gb,
        on=["positionID", "connectionType"],
        how='left')

    gb = temp.groupby(["advertiserID", "positionID"]).count().rename(
        columns={'count': 'advertiser_position_count'}).reset_index()
    gb = gb.loc[:, ["advertiserID", "positionID", "advertiser_position_count"]]
    temp = pd.merge(
        data[data["day"] == day],
        gb,
        on=["advertiserID", "positionID"],
        how='left')

    gb = temp.groupby(["gender", "positionID"]).count().rename(
        columns={'count': 'gender_position_count'}).reset_index()
    gb = gb.loc[:, ["gender", "positionID", "gender_position_count"]]
    temp = pd.merge(
        data[data["day"] == day], gb, on=["gender", "positionID"], how='left')
    reult = temp
    result = pd.concat([result, temp], axis=0)
result.fillna(0, inplace=True)
result.to_csv('count.csv')
print result
