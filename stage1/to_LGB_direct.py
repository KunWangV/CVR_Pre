from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

import pandas as pd
#from pyspark.ml.feature import OneHotEncoder as OHE
#import pyspark.sql.functions as F
import numpy as np
from sklearn.utils import shuffle  


from data import *
import pdb
import pickle

#from tqdm import tqdm
import os


def con(df, name1, name2, a1, a2):
    if a1 == 0 and a2 == 0:
        return df[name1] * df[name2]
    else:
        if a1 == 0 and a2 == 1:
            return df[name1] * df[name2]
        else:
            if a1 == 0 and a2 == 2:
                return pd.Series(df[name1].astype(
                    str) + (df[name2] / 0.05).astype(str)).astype('category').values.codes
            else:
                if a1 == 1 and a2 == 0:
                    return df[name1] * df[name2]
                else:
                    if a1 == 1 and a2 == 1:
                        return df[name1] + df[name2]
                    else:
                        if a1 == 2 and a2 == 0:
                            return pd.Series(df[name2].astype(
                                str) + (df[name1] / 0.05).astype(str)).astype('category').values.codes
                        else:
                            if a1 == 2 and a2 == 2:
                                return pd.Series(df[name2].astype(
                                    str) + (df[name1]).astype(str)).astype('category').values.codes
                            else:
                                return None


def to_LGBM(df_train, df_pre, test_days=2, if_CV=True):
    """
    cvt_positionID (5454,) 0.0 1.0
    cvt_userID (51,) 0.0 1.0
    cvt_creativeID (3880,) 0.0 1.0
    cvt_camgaignID (2377,) 0.0 1.0
    clickTime_day (14,) 17 30
    cvt_connectionType (65,) 0.00402765444628 0.0317665577059
    cnt_userID (87,) 0.0 111.0
    cvt_advertiserID (574,) 0.0 0.702702702703
    cnt_connectionType (65,) 2894.0 2726052.0
    cvt_sitesetID (39,) 0.0161008932448 0.0476605066562
    action_cate (88,) 0 117
    tt_cnt_appcate (121,) 0 250
    cnt_positionID (3059,) 0.0 351373.0
    appID (50,) 14 472
    cnt_appID (443,) 0.0 1765710.0
    appCategory (14,) 0 503
    cnt_advertiserID (625,) 0.0 1765710.0
    cvt_adID (3267,) 0.0 1.0
    age (81,) 0 80
    inst_app_installed (13,) 0 282777
    gender (3,) 0 2
    cvt_gender (39,) 0.0191151849487 0.0309996027044
    cnt_camgaignID (2109,) 0.0 554534.0
    cnt_appCategory (160,) 10.0 1765710.0
    advertiserID (89,) 1 91
    positionType (6,) 0 5
    cvt_appID (399,) 0.0 0.702702702703
    cnt_adID (2284,) 0.0 554534.0
    positionID (7219,) 1 7645
    hometown_p (35,) 0 34
    action_installed (2,) 0 1
    creativeID (6315,) 1 6582
    cnt_creativeID (2575,) 0.0 417728.0
    cvt_appPlatform (26,) 0.0235564738109 0.0285316369933
    clickTime_hour (24,) 0 23
    cvt_clickTime_week (8,) 0.0 0.0315482248123
    camgaignID (677,) 1 720
    action_cate_recent (22,) 0 28
    cvt_education (104,) 0.0198213640077 0.0352852852853
    inst_is_installed (2,) 0 1
    cvt_appCategory (149,) 0.0 0.702702702703
    sitesetID (3,) 0 2
    """
    feature_list1 = ['label',
                    'instanceID',
                    'clickTime_day',
                    'cvt_education'
                    'gender'
                    'cvt_clickTime_week',
                    'cnt_clickTime_week',
                    'clickTime_week',
                    'haveBaby',
                    'education',
                    'telecomsOperator',
                    'marriageStatus',
                    'appPlatform',
                    'hometown_c',
                    'cnt_education',
                    'inst_is_installed',
                    'residence_p',
                    'action_installed',
                    'cvt_haveBaby',
                    'cvt_hometown_c',
                    'clickTime_minute',
                    'cnt_gender',
                    'tt_is_installed',
                    'cnt_marriageStatus',
                    'cnt_residence_c',
                    'cvt_marriageStatus',
                    'cnt_haveBaby',
                    'cnt_adID',
                    'cnt_appPlatform',
                    'inst_cate_percent',
                    'adID',
                    'cvt_gender',
                    'cnt_residence_p',
                    'cnt_telecomsOperator',
                    'cnt_creativeID',
                    'inst_cnt_installed',
                    'cnt_connectionType',
                    'cnt_positionType',
                    'cvt_residence_c',
                    'cvt_appPlatform',
                    'cvt_residence_p',
                    'camgaignID',
                    'cnt_camgaignID',
                    'cvt_userID',
                    'cvt_telecomsOperator',
                    'inst_cnt_appcate',
                    'hometown_p',
                    'cnt_hometown_p',
                    'cvt_hometown_p',
                    'cvt_positionType',
                    'creativeID',
                    'action_cate_recent',
                    'positionID',
                    'clickTime_hour',
                    'inst_app_installed',
                    'userID',
                    'age',
                    'cvt_sitesetID',
                    'cnt_sitesetID',
                    'cnt_advertiserID',
                    'cnt_userID',
                    'cvt_connectionType',
                    'sitesetID',
                    'cvt_appID',
                    'connectionType',
                    'advertiserID',
                    'positionType',
                    'cnt_positionID',
                    'appID',
                    'cvt_adID',
                    'cvt_appCategory',
                    'cnt_appID',
                    'tt_cnt_appcate',
                    'cnt_appCategory',
                    'action_cate',
                    'appCategory',
                    'cvt_advertiserID',
                    'cvt_positionID',
                    'cvt_camgaignID',
                    'cvt_creativeID',
                    'user_day_click_cnt',
                    'user_cri_day_click_cnt',
                    # 'cri_day_click_cnt',
                    'cri_uuser_click_cnt',
                    'rpt_click_cnt',
                    'is_rpt_first_click',
                    'is_rpt_last_click',
                    'pre_cvt_userID',
                    'pre_cnt_userID',
                    'pre_cvt_creativeID',
                    'pre_cnt_creativeID',
                    'pre_cvt_positionID',
                    'pre_cnt_positionID',
                    'pre_cvt_adID',
                    'pre_cnt_adID',
                    'pre_cvt_camgaignID',
                    'pre_cnt_camgaignID',
                    'pre_cvt_advertiserID',
                    'pre_cnt_advertiserID',
                    'pre_cvt_appID',
                    'pre_cnt_appID',
                    'pre_cvt_sitesetID'	,
                    'pre_cnt_sitesetID'	,
                    'pre_cvt_appCategory',
                    'pre_cnt_appCategory',
                    'pre_cvt_appPlatform',
                    'pre_cnt_appPlatform',
                    'pre_cvt_education',
                    'pre_cnt_education',
                    'pre_cvt_gender',
                    'pre_cnt_gender',
                    'pre_cvt_haveBaby',
                    'pre_cnt_haveBaby',
                    'pre_cvt_marriageStatus',
                    'pre_cnt_marriageStatus',
                    'pre_cvt_positionType',
                    'pre_cnt_positionType',
                    'pre_cvt_hometown_c',
                    'pre_cnt_hometown_c',
                    'pre_cvt_hometown_p',
                    'pre_cnt_hometown_p',
                    'pre_cvt_residence_c',
                    'pre_cnt_residence_c',
                    'pre_cvt_residence_p',
                    'pre_cnt_residence_p',
                    'pre_cvt_telecomsOperator',
                    'pre_cnt_telecomsOperator',
                    'pre_cvt_connectionType',
                    'pre_cnt_connectionType',
                    'pre_cvt_clickTime_week',
                    'pre_cnt_clickTime_week',

                    ]


    feature_list=[
        'label',
        'instanceID',
        'clickTime_day',
        'cvt_positionID',
        'pre_cvt_userID',
        'cvt_creativeID',
        'cvt_userID',
        'cvt_camgaignID',
        'pre_cvt_camgaignID',
        'user_day_click_cnt',
        'pre_cvt_creativeID',
        'cvt_connectionType',
        'cnt_userID',
        'cvt_advertiserID',
        'cnt_connectionType',
        'is_rpt_first_click',
        'cvt_sitesetID',
        'action_cate',
        'pre_cnt_userID',
        'tt_cnt_appcate',
        'is_rpt_last_click',
        'cnt_appID',
        'pre_cvt_positionID',
        'cnt_advertiserID',
        'appCategory',
        'pre_cvt_connectionType',
        'cnt_positionID',
        'inst_app_installed',
        'gender',
        'pre_cnt_positionID',
        'cvt_adID',
        'age',
        'appID',
        'advertiserID',
        'pre_cvt_advertiserID',
        'user_cri_day_click_cnt',
        'positionType',
        'cnt_appCategory',
        'pre_cnt_camgaignID',
        'cvt_gender',
        'cri_day_click_cnt',
        'cnt_appPlatform',
        'tt_is_installed',
        'hometown_p',
        'pre_cvt_appID',
        'cvt_appID',
        'cvt_appCategory',
        'camgaignID',
        'cnt_camgaignID',
        'pre_cnt_advertiserID',
        'positionID',
        'action_installed',
        'pre_cvt_appCategory',
        'pre_cvt_adID',
        'sitesetID',
        'cnt_adID',
        'creativeID',
        'adID',
        'pre_cnt_sitesetID',
        'userID',
    ]
        # pre_cvt_userID|user_day_click_cnt
        # cvt_creativeID|cvt_positionID
        # cvt_connectionType|cvt_positionID
        # cvt_creativeID|pre_cvt_userID
        # cvt_positionID|pre_cvt_userID
        # cvt_positionID|cvt_positionID
        # cvt_positionID|pre_cvt_creativeID
        # pre_cvt_creativeID|pre_cvt_userID
        # cvt_positionID|pre_cvt_camgaignID
        # cvt_camgaignID|cvt_positionID
        # cnt_userID|cvt_positionID
        # cnt_userID|cvt_userID
        # cvt_camgaignID|cvt_creativeID
        # pre_cvt_camgaignID|pre_cvt_userID
        # is_rpt_first_click|user_day_click_cnt
        # cvt_creativeID|pre_cvt_camgaignID
        # cvt_advertiserID|cvt_connectionType
        # pre_cvt_camgaignID|pre_cvt_creativeID
        # cvt_userID|user_day_click_cnt
        # cvt_userID|is_rpt_last_click
        # cvt_creativeID|cvt_creativeID
        # cvt_camgaignID|pre_cvt_creativeID
        # cnt_connectionType|cvt_userID
        # cvt_positionID|cvt_sitesetID
        # cnt_userID|cvt_advertiserID
        # cvt_positionID|pre_cnt_positionID
        # cvt_camgaignID|cvt_userID


    LGBM_x = pd.concat([df_train, df_pre], axis=0)
    LGBM_x = LGBM_x.loc[:, feature_list]

    if 'cnt_sitesetID' in feature_list:
        LGBM_x.cnt_sitesetID = np.floor(LGBM_x.cnt_sitesetID / 10000)
    if 'cnt_appPlatform' in feature_list:
        LGBM_x.cnt_appPlatform =np.floor( LGBM_x.cnt_appPlatform / 10000)
    if 'cnt_education' in feature_list:
        LGBM_x.cnt_education = np.floor(LGBM_x.cnt_education / 10000)
    if 'cnt_gender' in feature_list:
        LGBM_x.cnt_gender = np.floor(LGBM_x.cnt_gender / 10000)
    if 'cnt_haveBaby' in feature_list:
        LGBM_x.cnt_haveBaby =np.floor( LGBM_x.cnt_haveBaby / 10000)
    if 'cnt_marriageStatus' in feature_list:
        LGBM_x.cnt_marriageStatus = np.floor(LGBM_x.cnt_marriageStatus / 10000)
    if 'cnt_hometown_c' in feature_list:
        LGBM_x.cnt_hometown_c =np.floor( LGBM_x.cnt_hometown_c / 10000)
    if 'cnt_hometown_p' in feature_list:
        LGBM_x.cnt_hometown_p =np.floor( LGBM_x.cnt_hometown_p / 10000)
    if 'cnt_residence_c' in feature_list:
        LGBM_x.cnt_residence_c =np.floor( LGBM_x.cnt_residence_c / 10000)
    if 'cnt_residence_p' in feature_list:
        LGBM_x.cnt_residence_p = np.floor(LGBM_x.cnt_residence_p / 10000)
    if 'cnt_telecomsOperator' in feature_list:
        LGBM_x.cnt_telecomsOperator =np.floor( LGBM_x.cnt_telecomsOperator / 10000)
    if 'cnt_clickTime_week' in feature_list:
        LGBM_x.cnt_clickTime_week =np.floor( LGBM_x.cnt_clickTime_week / 10000)
    if 'cnt_userID' in feature_list:
        LGBM_x.cnt_userID =np.floor( LGBM_x.cnt_userID / 10000)
    if 'cnt_connectionType' in feature_list:
        LGBM_x.cnt_connectionType =np.floor( LGBM_x.cnt_connectionType / 10000)
    if 'cnt_positionID' in feature_list:
        LGBM_x.cnt_positionID =np.floor( LGBM_x.cnt_positionID / 10000)
    if 'cnt_appID' in feature_list:
        LGBM_x.cnt_appID =np.floor( LGBM_x.cnt_appID / 10000)
    if 'cnt_advertiserID' in feature_list:
        LGBM_x.cnt_advertiserID =np.floor( LGBM_x.cnt_advertiserID / 10000)
    if 'cnt_camgaignID' in feature_list:
        LGBM_x.cnt_camgaignID =np.floor( LGBM_x.cnt_camgaignID / 10000)
    if 'cnt_appCategory' in feature_list:
        LGBM_x.cnt_appCategory =np.floor( LGBM_x.cnt_appCategory / 10000)
    if 'cnt_adID' in feature_list:
        LGBM_x.cnt_adID =np.floor( LGBM_x.cnt_adID / 10000)
    if 'cnt_creativeID' in feature_list:
        LGBM_x.cnt_creativeID =np.floor( LGBM_x.cnt_creativeID / 10000)

    if 'pre_cnt_sitesetID' in feature_list:
        LGBM_x.pre_cnt_sitesetID =np.floor( LGBM_x.pre_cnt_sitesetID / 10000)
    if 'pre_cnt_appPlatform' in feature_list:
        LGBM_x.pre_cnt_appPlatform =np.floor( LGBM_x.pre_cnt_appPlatform / 10000)
    if 'pre_cnt_education' in feature_list:
        LGBM_x.pre_cnt_education =np.floor( LGBM_x.pre_cnt_education / 10000)
    if 'pre_cnt_gender' in feature_list:
        LGBM_x.pre_cnt_gender =np.floor( LGBM_x.pre_cnt_gender / 10000)
    if 'pre_cnt_haveBaby' in feature_list:
        LGBM_x.pre_cnt_haveBaby =np.floor( LGBM_x.pre_cnt_haveBaby / 10000)
    if 'pre_cnt_marriageStatus' in feature_list:
        LGBM_x.pre_cnt_marriageStatus =np.floor( LGBM_x.pre_cnt_marriageStatus / 10000)
    if 'pre_cnt_hometown_c' in feature_list:
        LGBM_x.pre_cnt_hometown_c =np.floor( LGBM_x.pre_cnt_hometown_c / 10000)
    if 'pre_cnt_hometown_p' in feature_list:
        LGBM_x.pre_cnt_hometown_p =np.floor( LGBM_x.pre_cnt_hometown_p / 10000)
    if 'pre_cnt_residence_c' in feature_list:
        LGBM_x.pre_cnt_residence_c =np.floor( LGBM_x.pre_cnt_residence_c / 10000)
    if 'pre_cnt_residence_p' in feature_list:
        LGBM_x.pre_cnt_residence_p =np.floor( LGBM_x.pre_cnt_residence_p / 10000)
    if 'pre_cnt_telecomsOperator' in feature_list:
        LGBM_x.pre_cnt_telecomsOperator =np.floor( LGBM_x.pre_cnt_telecomsOperator / 10000)
    if 'pre_cnt_clickTime_week' in feature_list:
        LGBM_x.pre_cnt_clickTime_week =np.floor( LGBM_x.pre_cnt_clickTime_week / 10000)
    if 'pre_cnt_userID' in feature_list:
        LGBM_x.pre_cnt_userID =np.floor( LGBM_x.pre_cnt_userID / 10000)
    if 'pre_cnt_connectionType' in feature_list:
        LGBM_x.pre_cnt_connectionType =np.floor( LGBM_x.pre_cnt_connectionType / 10000)
    if 'pre_cnt_positionID' in feature_list:
        LGBM_x.pre_cnt_positionID =np.floor( LGBM_x.pre_cnt_positionID / 10000)
    if 'pre_cnt_appID' in feature_list:
        LGBM_x.pre_cnt_appID =np.floor( LGBM_x.pre_cnt_appID / 10000)
    if 'pre_cnt_advertiserID' in feature_list:
        LGBM_x.pre_cnt_advertiserID =np.floor( LGBM_x.pre_cnt_advertiserID / 10000)
    if 'pre_cnt_camgaignID' in feature_list:
        LGBM_x.pre_cnt_camgaignID =np.floor( LGBM_x.pre_cnt_camgaignID / 10000)
    if 'pre_cnt_appCategory' in feature_list:
        LGBM_x.pre_cnt_appCategory =np.floor( LGBM_x.pre_cnt_appCategory / 10000)
    if 'pre_cnt_adID' in feature_list:
        LGBM_x.pre_cnt_adID =np.floor( LGBM_x.pre_cnt_adID / 10000)
    if 'pre_cnt_creativeID' in feature_list:
        LGBM_x.pre_cnt_creativeID =np.floor( LGBM_x.pre_cnt_creativeID / 10000)

    if 'appID' in feature_list:
        LGBM_x['appID'] = pd.Series(
            LGBM_x['appID']).astype('category').values.codes
    if 'appCategory' in feature_list:
        LGBM_x['appCategory'] = pd.Series(
            LGBM_x['appCategory']).astype('category').values.codes
    if 'age' in feature_list:
        LGBM_x.age = LGBM_x.age / 5


    # pre_cvt_userID|user_day_click_cnt
    LGBM_x['pre_cvt_userID_user_day_click_cnt'] = con(LGBM_x, 'pre_cvt_userID', 'user_day_click_cnt', 0, 1)
    # cvt_creativeID|cvt_positionID
    LGBM_x['cvt_creativeID_cvt_positionID'] = con(LGBM_x, 'cvt_creativeID', 'cvt_positionID', 0, 0)
    # cvt_connectionType|cvt_positionID
    LGBM_x['cvt_connectionType_cvt_positionID'] = con(LGBM_x, 'cvt_connectionType', 'cvt_positionID', 0, 0)
    # cvt_creativeID|pre_cvt_userID
    LGBM_x['cvt_creativeID_pre_cvt_userID'] = con(LGBM_x, 'cvt_creativeID', 'pre_cvt_userID', 0, 0)
    # cvt_positionID|pre_cvt_userID
    LGBM_x['cvt_positionID_pre_cvt_userID'] = con(LGBM_x, 'cvt_positionID', 'pre_cvt_userID', 0, 0)
    # cvt_positionID|cvt_positionID
    LGBM_x['cvt_positionID_cvt_positionID'] = con(LGBM_x, 'cvt_positionID', 'cvt_positionID', 0, 0)
    # cvt_positionID|pre_cvt_creativeID
    LGBM_x['cvt_positionID_pre_cvt_creativeID'] = con(LGBM_x, 'cvt_positionID', 'pre_cvt_creativeID', 0,0)
    # pre_cvt_creativeID|pre_cvt_userID
    LGBM_x['pre_cvt_creativeID_pre_cvt_userID'] = con(LGBM_x, 'pre_cvt_creativeID', 'pre_cvt_userID', 0, 0)
    # cvt_positionID|pre_cvt_camgaignID
    LGBM_x['cvt_positionID_pre_cvt_camgaignID'] = con(LGBM_x, 'cvt_positionID', 'pre_cvt_camgaignID', 0, 0)
    # cvt_camgaignID|cvt_positionID
    LGBM_x['cvt_camgaignID_cvt_positionID'] = con(LGBM_x, 'cvt_camgaignID', 'cvt_positionID', 0, 0)
    # cnt_userID|cvt_positionID
    LGBM_x['cnt_userID_cvt_positionID'] = con(LGBM_x, 'cnt_userID', 'cvt_positionID', 1, 0)

    # cnt_userID|cvt_positionID
    LGBM_x['cnt_userID_cvt_positionID']=con(LGBM_x,'cnt_userID','cvt_positionID',1,0)

    # cnt_userID|cvt_userID
    LGBM_x['cnt_userID_cvt_userID']=con(LGBM_x,'cnt_userID','cvt_userID',1,0)

    # cvt_camgaignID|cvt_creativeID
    LGBM_x['cvt_camgaignID_cvt_creativeID']=con(LGBM_x,'cvt_camgaignID','cvt_creativeID',0,0)

    # pre_cvt_camgaignID|pre_cvt_userID
    LGBM_x['pre_cvt_camgaignID_pre_cvt_userID']=con(LGBM_x,'pre_cvt_camgaignID','pre_cvt_userID',0,0)

    # is_rpt_first_click|user_day_click_cnt
    LGBM_x['is_rpt_first_click_user_day_click_cnt']=con(LGBM_x,'is_rpt_first_click','user_day_click_cnt',2,2)

    # cvt_creativeID|pre_cvt_camgaignID
    LGBM_x['cvt_creativeID_pre_cvt_camgaignID']=con(LGBM_x,'cvt_creativeID','pre_cvt_camgaignID',0,0)

    # cvt_advertiserID|cvt_connectionType
    LGBM_x['cvt_advertiserID_cvt_connectionType']=con(LGBM_x,'cvt_advertiserID','cvt_connectionType',0,0)

    # pre_cvt_camgaignID|pre_cvt_creativeID
    LGBM_x['pre_cvt_camgaignID_pre_cvt_creativeID']=con(LGBM_x,'pre_cvt_camgaignID','pre_cvt_creativeID',0,0)

    # cvt_userID|user_day_click_cnt
    LGBM_x['cvt_userID_user_day_click_cnt']=con(LGBM_x,'cvt_userID','user_day_click_cnt',0,1)

    # cvt_userID|is_rpt_last_click
    LGBM_x['cvt_userID_is_rpt_last_click']=con(LGBM_x,'cvt_userID','is_rpt_last_click',0,2)

    # cvt_creativeID|cvt_creativeID
    LGBM_x['cvt_creativeID_cvt_creativeID']=con(LGBM_x,'cvt_creativeID','cvt_creativeID',0,0)

    # cvt_camgaignID|pre_cvt_creativeID
    LGBM_x['cvt_camgaignID_pre_cvt_creativeID']=con(LGBM_x,'cvt_camgaignID','pre_cvt_creativeID',0,0)

    # cnt_connectionType|cvt_userID
    LGBM_x['cnt_connectionType_cvt_userID']=con(LGBM_x,'cnt_connectionType','cvt_userID',1,0)

    # cvt_positionID|cvt_sitesetID
    LGBM_x['cvt_positionID_cvt_sitesetID']=con(LGBM_x,'cvt_positionID','cvt_sitesetID',0,0)

    # cnt_userID|cvt_advertiserID
    LGBM_x['cnt_userID_cvt_advertiserID']=con(LGBM_x,'cnt_userID','cvt_advertiserID',1,0)

    # cvt_positionID|pre_cnt_positionID
    LGBM_x['cvt_positionID_pre_cnt_positionID']=con(LGBM_x,'cvt_positionID','pre_cnt_positionID',0,1)

    # cvt_camgaignID|cvt_userID
    LGBM_x['cvt_camgaignID_cvt_userID']=con(LGBM_x,'cvt_camgaignID','cvt_userID',0,0)

    LGBM_x['instanceID'].fillna(-1, inplace=True)
    LGBM_x.fillna(0, inplace=True)
    LGBM_x = shuffle(LGBM_x)  
    pre_x = LGBM_x.loc[LGBM_x['instanceID'] > 0].copy()
    pre_x.sort_values('instanceID', inplace=True)
    inst_id = pre_x['instanceID'].copy().values
    pre_x.drop(['instanceID'], axis=1, inplace=True)
    pre_x.drop(['label', 'clickTime_day'], axis=1, inplace=True)

    LGBM_x.drop(['instanceID'], axis=1, inplace=True)
    if not if_CV:
        train_x = LGBM_x.loc[(
            (LGBM_x['clickTime_day'] <= (30 - test_days))), :].copy()
        test_x = LGBM_x.loc[(LGBM_x['clickTime_day'] <= 30) &
                            (LGBM_x['clickTime_day'] > (30 - test_days)), :].copy()
        test_x1 = LGBM_x.loc[(LGBM_x['clickTime_day'] <= 29) &
                             (LGBM_x['clickTime_day'] > (30 - test_days)), :].copy()
        # print 'pre_x.shape:'
        # print pre_x.shape

        # print train_x.columns
        train_y = np.round(train_x['label']).astype(int).values
        train_x.drop(['label', 'clickTime_day'], 1, inplace=True)

        test_y = np.round(test_x['label']).astype(int).values
        test_x.drop(['label', 'clickTime_day'], 1, inplace=True)

        test_y1 = np.round(test_x1['label']).astype(int).values
        test_x1.drop(['label', 'clickTime_day'], 1, inplace=True)

        train_x = train_x.values
        test_x = test_x.values
        test_x1 = test_x1.values
        pre_x = pre_x.values

        return train_x, train_y, test_x, test_y, test_x1, test_y1, pre_x, inst_id
    else:
        x = LGBM_x.loc[LGBM_x['clickTime_day'] <= 30, :].copy()
        groups = x.clickTime_day.values
        y = np.round(x['label']).astype(int).values
        x.drop(['label', 'clickTime_day'], 1, inplace=True)
        x = x.values
        pre_x = pre_x.values

        return x, y, groups, pre_x, inst_id
