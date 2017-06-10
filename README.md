## data structure

``` python
test.csv
['instanceID' 'label' 'clickTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator'] (3321748, 8)
   instanceID  label  clickTime  creativeID    userID  positionID    connectionType  telecomsOperator  
0           1     -1   31000000       19553  11856541        4522          1                 1  
=====================================================
column: instanceID           unique size:    3321748 max:    3321748 min:          1
column: label                unique size:          1 max:         -1 min:         -1
column: clickTime            unique size:      86391 max:   31235959 min:   31000000
column: creativeID           unique size:      17169 max:      51753 min:          4
column: userID               unique size:    2496619 max:   20062200 min:          3
column: positionID           unique size:      12902 max:      21922 min:          2
column: connectionType       unique size:          5 max:          4 min:          0
column: telecomsOperator     unique size:          4 max:          3 min:          0
=====================================================
   
train.csv
['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator'] (37912ls916, 8)
   label  clickTime  conversionTime  creativeID   userID  positionID  connectionType  telecomsOperator  
0      0   16231202             NaN       42905  3143312        3322     1                 3 
=====================================================
column: label                unique size:          2 max:          1 min:          0
column: clickTime            unique size:    1209270 max:   30235959 min:   16231202
column: conversionTime       unique size:     586367 max: 30235959.0 min: 17000129.0
column: creativeID           unique size:      48836 max:      51754 min:          1
column: userID               unique size:   18903884 max:   20062201 min:          1
column: positionID           unique size:      21488 max:      21922 min:          1
column: connectionType       unique size:          5 max:          4 min:          0
column: telecomsOperator     unique size:          4 max:          3 min:          0
=====================================================

ad.csv
['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform'] (51754, 6)
   creativeID   adID  camgaignID  advertiserID  appID  appPlatform
0       28934  29217        3346           103    369            1
=====================================================
column: creativeID           unique size:      51754 max:      51754 min:          1
column: adID                 unique size:      31467 max:      31467 min:          1
column: camgaignID           unique size:       6940 max:       6940 min:          1
column: advertiserID         unique size:        659 max:        659 min:          1
column: appID                unique size:        479 max:        479 min:          1
column: appPlatform          unique size:          2 max:          2 min:          1
=====================================================

app_categories.csv
['appID' 'appCategory'] (433269, 2)
   appID  appCategory
0      1          407
=====================================================
column: appID                unique size:     433269 max:     433269 min:          1
column: appCategory          unique size:         31 max:        503 min:          0
=====================================================

position.csv
['positionID' 'sitesetID' 'positionType'] (21922, 3)
   positionID  sitesetID  positionType
0        6315          0             1
=====================================================
column: positionID           unique size:      21922 max:      21922 min:          1
column: sitesetID            unique size:          3 max:          2 min:          0
column: positionType         unique size:          6 max:          5 min:          0
=====================================================

user_app_actions.csv
['userID' 'installTime' 'appID'] (38819295, 3)
   userID  installTime  appID
0       1     18203243    933
=====================================================
column: userID               unique size:    5224685 max:   20062201 min:          1
column: installTime          unique size:    2223792 max:   30235959 min:    1000000
column: appID                unique size:     210924 max:     433267 min:        354
=====================================================

user.csv
['userID' 'age' 'gender' 'education' 'marriageStatus' 'haveBaby' 'hometown' 'residence'] (20062201, 8)
   userID  age  gender  education  marriageStatus  haveBaby  hometown  residence  
0       1   42       1          0               2         0       512     503
=====================================================
column: userID               unique size:   20062201 max:   20062201 min:          1
column: age                  unique size:         81 max:         80 min:          0
column: gender               unique size:          3 max:          2 min:          0
column: education            unique size:          8 max:          7 min:          0
column: marriageStatus       unique size:          4 max:          3 min:          0
column: haveBaby             unique size:          7 max:          6 min:          0
column: hometown             unique size:        365 max:       3401 min:          0
column: residence            unique size:        405 max:       3401 min:          0
=====================================================

user_installedapps.csv
['userID' 'appID'] (575832463, 2)
   userID  appID
0       1    357
=====================================================
column: userID               unique size:    9850468 max:   20062197 min:          1
column: appID                unique size:     377296 max:     433269 min:        354
=====================================================
```