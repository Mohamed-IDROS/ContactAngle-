# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:53:11 2020

@author: Mohamed Nazmi Idros 
"""

from CAanalysis import CA_analysis
import glob
import pandas as pd
#%%
#
#list1=[]
#summary=pd.DataFrame(columns=['name'])
df=pd.DataFrame(columns=['name','Circle left angle','Circle right angle','Circle average angle','Elipsse left angle','Elipsse right angle', 'Elipsse average angle'])
i=0
for filename in glob.glob('rawdata/*'):
#    print(filename)
    al_c,ar_c, av_c, al_e, ar_e, av_e = CA_analysis (filename)
    df.loc[i]=[filename.replace('rawdata\\','').replace('.bmp','').replace('.jpg','').replace('.tif',''), al_c,ar_c, av_c, al_e, ar_e, av_e ]
    i+=1

df.to_excel('processdata/data summary.xlsx')
