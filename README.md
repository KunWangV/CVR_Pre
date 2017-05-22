# 数据
## 每个文件的样例数据
```code
广告 ==> ad.csv <==
creativeID,adID,camgaignID,advertiserID,appID,appPlatform
    creativeID (6582L,)
    adID (3616L,)
    camgaignID (720L,)
    advertiserID (91L,)
    appID (50L,)
    appPlatform (2L,)
4079,2318,147,80,14,2

APP类别==> app_categories.csv <==
appID,appCategory(共28个类别)
14,2

广告位==> position.csv <==
positionID,sitesetID,positionType
2150,1,0

用户安装流水 ==> user_app_actions.csv <==
userID,installTime,appID
1,182032,933

用户==> user.csv <==
userID,age,gender,education,marriageStatus,haveBaby,hometown(365),residence(400-二级编码)
1,42,1,0,2,0,512,503

用户已经安装的APP==> user_installedapps.csv <==
userID,appID
1,357

测试数据==> test.csv <==
instanceID,label,clickTime,creativeID,userID,positionID,connectionType,telecomsOperator
1,-1,310000,3745,1164848,3451,1,3

训练数据==> train.csv <==
label,clickTime,conversionTime,creativeID,userID,positionID,connectionType,telecomsOperator
    label (2L,)
    clickTime (20160L,)
    conversionTime (18207L,)
    creativeID (6315L,)
    userID (2595627L,)
    positionID (7219L,)
    connectionType (5L,)
    telecomsOperator (4L,)
0,170000,,3089,2798058,293,1,1

提交==> submission.csv <==
instanceID,prob
1,0.353741138898
```

# 提交记录
- 2015/5/10 随机: 0.116798
- 2015/5/22 全0：0.862443
- 2015/5/22 全1：33.6771
