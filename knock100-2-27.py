#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:42:23 2023

@author: masayasusaito
"""

import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データの読み込み
data = pd.read_excel('data/1-2-2020.xlsx',skiprows=4,header=None)
data.drop(data.index[-4:],inplace=True)       
data.tail(6)

# columnの抽出と付与
col_data = pd.read_excel('data/1-2-2020.xlsx',skiprows=1,header=None)
col_data = col_data.head(3)
col_data

# セル結合の影響除去（NaN欠損の処理）;行方向
col_data.iloc[1,1:].fillna(col_data.iloc[0,1:],inplace=True)
col_data.iloc[1,1:] = col_data.iloc[1,1:].str.replace('発電所','')

# 基本形は、 df[条件範囲].fillna(置き換えたい値)

# dfの中の欠損値を（条件範囲のもの）：列方向
for i in col_data.columns:
    if i < col_data.columns.max():# 最大列数を超えない対策
        col_data[i+1].fillna(col_data[i],inplace=True)

col_data
col_data.replace('〔バイオマス〕','バイオマス',inplace=True)
col_data.replace('〔廃棄物〕','廃棄物',inplace=True)

# カラム名の修正
cols = []
for i in col_data.columns:
    tg_col = '_'.join(list(col_data[i].dropna())) # 1列目を足しこむ
    cols.append(tg_col)
cols

# カラムの代入

data.columns = cols

# 全テータの読み込み
xl = pd.ExcelFile('data/1-2-2020.xlsx')
sheets = xl.sheet_names

datas=[]
for sheet in sheets:
    data = xl.parse(sheets[0],skiprows=4,header=None)
    data.drop(data.tail(4).index, inplace=True)
    data.columns = cols
    data['年月']=sheet
    datas.append(data)
    
#　データの結合（Conacat ）
datas=pd.concat(datas,ignore_index=True)

# データの修正
datas['火力発電所_火力_発電所数'] = datas['火力発電所_火力_発電所数']- datas['新エネルギー等発電所_バイオマス_発電所数']- datas['新エネルギー等発電所_廃棄物_発電所数']
                                                                                      
datas['火力発電所_火力_最大出力計'] = datas['火力発電所_火力_最大出力計']-datas['新エネルギー等発電所_バイオマス_最大出力計']-datas['新エネルギー等発電所_廃棄物_最大出力計']
datas.head ()

# データのカラムの絞り込み

datas.drop(['合計_合計_発電所数','合計_合計_最大出力計','新エネルギー等発電所_計_発電所数','新エネルギー等発電所_計_最大出力計'],axis=1,inplace=True)

#　縦持ちデータへの変換 (melt)
 
datas_v = pd.melt(datas,id_vars=['都道府県','年月'], var_name='変数名',value_name='値')
datas_v.head()

#  変数名を再度分離する
var_data = datas_v['変数名'].str.split('_',expand=True)
var_data.head()

# カラム名として分離した変数を　datas_vと　結合

var_data.columns = ['発電所種別','発電種別','項目']
datas_v = pd.concat([datas_v, var_data], axis=1)

# 不要となった、変数名の列を削除
datas_v.drop(['変数名'], axis=1, inplace=True)
datas_v.head()

#　********* knock 27　*************


# 発電データの読み込み 28
col_ca_data = pd.read_excel('data/2-2-2020.xlsx',skiprows=1,header=None)
col_ca_data = col_ca_data.head(3)     

col_ca_data.iloc[1,1:].fillna(col_ca_data.iloc[0,1:],inplace=True)
col_ca_data.iloc[1,1:] = col_ca_data.iloc[1,1:].str.replace('発電所','')

for i in col_ca_data.columns:
    if i < col_ca_data.columns.max():
        col_ca_data[i+1].fillna(col_ca_data[i],inplace=True)

col_ca_data.replace('〔バイオマス〕','バイオマス',inplace=True)
col_ca_data.replace('〔廃棄物〕','廃棄物',inplace=True)

cols_ca = []
for i in col_ca_data.columns:
    tg_col = '_'.join(list(col_ca_data[i].dropna()))
    cols_ca.append(tg_col)
cols_ca

# 結合の実施  2-23
xl_ca = pd.ExcelFile('data/2-2-2020.xlsx')
sheets = xl_ca.sheet_names
ca_datas=[]
for sheet in sheets:
    capacity_data = xl_ca.parse(sheet,skiprows=4,header=None)
    capacity_data = capacity_data.head(47)
    capacity_data.columns = cols_ca
    capacity_data['年月']=sheet
    ca_datas.append(capacity_data)


ca_datas = pd.concat(ca_datas,ignore_index=True)
ca_datas.head()

# データ修正 2-24
ca_datas['火力発電所_火力_電力量'] = ca_datas['火力発電所_火力_電力量']- ca_datas['新エネルギー等発電所_バイオマス_電力量']- ca_datas['新エネルギー等発電所_廃棄物_電力量']
ca_datas.drop(['合計_合計_電力量','新エネルギー等発電所_計_電力量'],axis=1,inplace=True)

ca_datas_v = pd.melt(ca_datas,id_vars=['都道府県','年月'], var_name='変数名',value_name='値')
ca_datas_v.head()
ca_var_data = ca_datas_v['変数名'].str.split('_',expand=True)
ca_var_data.columns = ['発電所種別','発電種別','項目']
ca_datas_v = pd.concat([ca_datas_v,ca_var_data],axis=1) # this!!
ca_datas_v.drop(['変数名'],axis=1, inplace=True)
ca_datas_v.head()

# 総仕上げ 29
from IPython.display import display

import os

datas_v_all = pd.concat([datas_v,ca_datas_v],ignore_index=True)

data_file='datas_v_all.xlsx'
out_dir=('data/')

datas_v_all.to_excel(os.path.join(out_dir,data_file),index=False)



display(datas_v_all.head())
display(datas_v_all.tail())

pd.pivot_table(datas_v_all.loc[datas_v_all['年月']=='2020.4'],index='発電所種別',columns='項目',values='値',aggfunc='sum')

# *************
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os
import japanize_matplotlib
import seaborn as sns

data_file='datas_v_all.xlsx'
out_dir=('data/')
datas_v_all2 = pd.read_excel(os.path.join(out_dir,data_file),index_col=None)

datas_v_all == datas_v_all2

datas_v_all.dtypes
datas_v_all2.dtypes

## ノック３０：データ分布をヒストグラムで可視化してみよう

plt.figure(figsize=(20, 10))
sns.histplot(datas_v_all2.loc[datas_v_all['項目']=='発電所数'])

fig, axes = plt.subplots(1, 3, figsize=(30, 10))

viz_data = datas_v_all.loc[datas_v_all['値']!=0]

viz_data = datas_v_all
sns.histplot(viz_data.loc[viz_data['項目']=='発電所数'], ax=axes[0])
sns.histplot(viz_data.loc[viz_data['項目']=='最大出力計'], ax=axes[1])
sns.histplot(viz_data.loc[viz_data['項目']=='電力量'], ax=axes[2])

## ノック３１：データ分布を箱ひげ図で可視化してみよう



# viz_data = datas_v_all.loc[(datas_v_all['項目']=='電力量')&(datas_v_all['値']!=0)]

viz_data = datas_v_all[['発電種別','値']].loc[(datas_v_all['項目']=='電力量')&(datas_v_all['値']!=0)]
plt.figure(figsize=(30, 10))


p= sns.boxplot(x=viz_data['発電種別'], y=viz_data['値'])
p.set_title('電力量',fontsize=25)


viz_data = datas_v_all[['発電種別','値']].loc[(datas_v_all['項目']=='電力量')&(datas_v_all['年月']=='2021.1')]
viz_data = viz_data.groupby('発電種別', as_index=False).sum()
viz_data


sns.barplot(x=viz_data['発電種別'], y=viz_data['値'])


## ノック３３：先月の発電量とあわせて可視化してみよう


# 基本系

# df2 = df1 [ ['使いたいカラム','clolumn2','',...] ] .loc [ (df1==条件1) &| (df1=>条件２)...]
# plt(or sns).graph_name(x=df2[' ',' '], y=df2[''], pram1=.., pram2=...)


viz_data = datas_v_all[['発電種別','年月','値']].loc[(datas_v_all['項目']=='電力量')&((datas_v_all['年月']=='2020.12')|(datas_v_all['年月']=='2021.1'))]
viz_data = viz_data.groupby(['発電種別','年月'],as_index=False).sum()
viz_data.head()


# viz_data = viz_data.loc[(viz_data['年月']=='2020.12')|(viz_data['年月']=='2021.1')]
sns.barplot(x=viz_data['発電種別'],y=viz_data['値'],hue=viz_data['年月'])

## ノック３４：電力の時系列変化を可視化してみよう


plt.figure(figsize=(15, 5))
viz_data = datas_v_all[['発電種別','年月','値']].loc[(datas_v_all['項目']=='電力量')]

viz_data = viz_data.groupby('年月',as_index=False).sum()
# 年月でソートする
viz_data['年月'] = pd.to_datetime(viz_data['年月'])

sns.lineplot(x=viz_data['年月'], y=viz_data["値"])


# 発電種類別の月推移 34の続き
plt.figure(figsize=(15, 5))
viz_data = datas_v_all[['発電種別','年月','値']].loc[(datas_v_all['項目']=='電力量')]
viz_data = viz_data.groupby(['発電種別','年月'],as_index=False).sum()
viz_data['年月'] = pd.to_datetime(viz_data['年月'])
sns.lineplot(x=viz_data['年月'], y=viz_data["値"], hue=viz_data['発電種別'])


## ノック３５：電力の割合を可視化してみよう


viz_data = datas_v_all.loc[(datas_v_all['項目']=='電力量')&(datas_v_all['年月']=='2021.1')]
viz_data = viz_data[['発電種別','値']].groupby('発電種別').sum()
viz_data['割合'] = viz_data['値'] / viz_data['値'].sum()
viz_data



viz_data.T.loc[['割合']].plot(kind='bar', stacked=True)


## ノック３６：電力量の多い都道府県を比較してみよう


viz_data = datas_v_all.loc[datas_v_all['項目']=='電力量']
viz_data = viz_data[['都道府県','値']].groupby('都道府県', as_index=False).sum()

viz_data.sort_values('値', inplace=True, ascending=False) # sort_value
viz_data.head(5)


plt.figure(figsize=(15, 5))
viz_data = datas_v_all[['都道府県','年月','値']].loc[(datas_v_all['項目']=='電力量')&((datas_v_all['都道府県']=='神奈川県')|(datas_v_all['都道府県']=='千葉県'))]
viz_data = viz_data.groupby(['年月', '都道府県'],as_index=False).sum()
viz_data['年月'] = pd.to_datetime(viz_data['年月'])
sns.lineplot(x=viz_data['年月'], y=viz_data["値"], hue=viz_data['都道府県'])


viz_data_num = datas_v_all[['都道府県','年月','値']].loc[(datas_v_all['項目']=='発電所数')&((datas_v_all['都道府県']=='神奈川県')|(datas_v_all['都道府県']=='千葉県'))]
viz_data_num = viz_data_num.groupby(['年月', '都道府県'],as_index=False).sum()
viz_data_num['年月'] = pd.to_datetime(viz_data_num['年月'])
viz_data.rename(columns={'値':'電力量'}, inplace=True)
viz_data_num.rename(columns={'値':'発電所数'}, inplace=True)
viz_data_join = pd.merge(viz_data, viz_data_num, on=['年月', '都道府県'], how='left')
viz_data_join.head()



sns.relplot(x=viz_data_join['年月'],  y=viz_data_join['電力量'], 
            hue=viz_data_join['都道府県'], size=viz_data_join['発電所数'],
            alpha=0.5, height=5, aspect=2)



## ノック３７：都道府県、年月別の電力量を可視化してみよう



viz_data = datas_v_all[['都道府県','年月','値']].loc[datas_v_all['項目']=='電力量']
viz_data = viz_data.groupby(['年月', '都道府県'],as_index=False).sum()
viz_data['年月'] = pd.to_datetime(viz_data['年月']).dt.date

viz_data = viz_data.pivot_table(values='値', columns='年月', index='都道府県')
viz_data.head(5)


plt.figure(figsize=(10,10))
sns.heatmap(viz_data)


## ノック３８：変数の関係性を可視化してみよう


viz_data = datas.drop(['都道府県','年月'],axis=1)
viz_data.head(5)



sns.scatterplot(x=viz_data['水力発電所_水力_発電所数'], y=viz_data['水力発電所_水力_最大出力計'])


sns.jointplot(x=viz_data['水力発電所_水力_発電所数'], y=viz_data['水力発電所_水力_最大出力計'])


sns.pairplot(viz_data.iloc[:,0:4])

viz_data
viz_data.dtypes

viz2= viz_data.iloc[:,0:4]
viz2.dtypes

## ノック３９：データを整形してExcel形式で出力しよう



output = datas_v_all.pivot_table(values='値', columns='項目', index=['年月','都道府県'], aggfunc='sum')
output.head()



output.to_excel('data/summary_data.xlsx')

## ノック４０：シート別にExcelデータを出力しよう




import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import os
data_file='datas_v_all.xlsx'
out_dir=('data/')
datas_v_all= pd.read_excel(os.path.join(out_dir,data_file),index_col=None)

target = '北海道'
tmp = datas_v_all.loc[datas_v_all['都道府県']==target]
tmp = tmp.pivot_table(values='値', columns=['発電種別'], index=['年月','項目'], aggfunc=np.mean)

tmp.head(5)



datas_v_all['都道府県'] # 12689県になる
datas_v_all['都道府県'].unique()
len(datas_v_all['都道府県'].unique()) #47県になる

# ExcelWriterオブジェクトを使うと、複数のpandas.DataFrameオブジェクトを別々のシートに書き出すことが可能
# pandas.ExcelWriter()にパスを指定してExcelWriterオブジェクトを生成し、to_excel()メソッドの第一引数に指定する。
# withブロックを使うとwriter.save(), writer.close()を呼ぶ必要がないので楽

# with pd.ExcelWriter('data/dst/pandas_to_excel_multi.xlsx') as writer:
#    df.to_excel(writer, sheet_name='sheet1')
#    df2.to_excel(writer, sheet_name='sheet2')

with pd.ExcelWriter('data/detail_data.xlsx', mode='w') as writer:
    for target in datas_v_all['都道府県'].unique():
        tmp = datas_v_all.loc[datas_v_all['都道府県']==target]
        tmp = tmp.pivot_table(values='値', columns=['発電種別','項目'], index=['年月'], aggfunc='sum')
        tmp.to_excel(writer, sheet_name=target)




