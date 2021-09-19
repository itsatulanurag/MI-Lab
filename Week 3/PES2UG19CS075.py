import numpy as np
import pandas as pd


def get_entropy_of_dataset(df):
    elements=df.play.unique()
    entropy=0
    for i in elements:
        probability=df.play.value_counts()[i]/len(df.play)
        entropy-=probability*np.log2(probability)
    return entropy


def get_avg_info_of_attribute(df, attribute):
    columns = df.keys()[-1]
    elements = df[columns].unique()
    var = df[attribute].unique()
    entropy = 0
    for i in var:
        temp = 0
        for target_variable in elements:
            num = len(df[attribute][df[attribute]==i]
                        [df[columns] ==target_variable])
            deno = len(df[attribute][df[attribute]==i])
            p = num/(deno+np.finfo(float).eps)
            temp += -p*np.log2(p+np.finfo(float).eps)
        q = deno/len(df)
        entropy += q*temp
    return entropy


def get_information_gain(df, attribute):
    attr_entropy=get_avg_info_of_attribute(df,attribute)
    entropy=get_entropy_of_dataset(df)
    return(entropy-attr_entropy)


def get_selected_attribute(df):
    info_gains={}
    col=''
    max_info_gain=float("-inf")
    for attr in df.columns[:-1]:
        info_gain_attr=get_information_gain(df,attr)
        if(info_gain_attr>max_info_gain):
            col=attr
            max_info_gain=info_gain_attr
        info_gains[attr]=info_gain_attr
    return(info_gains,col)