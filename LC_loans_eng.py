#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:54:46 2019

@author: eran
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


directory_path='Desktop/Galvanize-Resources/innterview_Qs/ThetaRay_takehome'

# 1. Let's read in the data 
raw=pd.read_csv(os.path.join(directory_path,'LoanStats3c.csv'))
raw.head(2)

# to plot, turn the following statement to True
if 1==0: 
    plt.subplots(figsize=(16,3))
    sns.heatmap(raw.isnull(),cbar=False)
    iii=plt.title('nulls in raw dataset')
    
'''Features to drop/keep
After LC online research, and using the dictionary given. the columns (fields) that should stay and ones 
that should be removed are given in the following summary I created:'''

col_rel=pd.read_csv('/Users/eran/Desktop/Galvanize-Resources/innterview_Qs/ThetaRay_takehome/col_relevant.csv',)
col_rel=col_rel.iloc[:,0:2]
col_rel=list(col_rel.iloc[:,0][col_rel['0']==1])
col_rel ## list of the columns to keep

# 2. create labels (response) from the data
''' Find labels to "predict" based on business opportunity for Lending Club '''

'''Percent Projected to be charged off (by loan_status):
From : https://www.lendingclub.com/info/demand-and-credit-profile.action
22% of Grace Period
50% of late(16-30 days)
75% of late(31-120 days)
72% of Default
seems that Default and Late have similar prospects. let's include " a bit late" 
in the fully paid group and the rest in the Default group'''

# creating 3 classes: Current = 2, Fully Paid = 0, Deafault =1
print('data has now a response variable - "Loan_response" with 3 classes:')
print('Current = 2, Fully Paid = 0, Deafault =1')
raw_lab=raw.copy()
raw_lab['Loan_response']=raw_lab['loan_status'].apply(lambda x: 1 if (x=='Late (31-120 days)') or (x=='Charged Off') or (x=='Default') else (2 if (x=='Current') else 0))

# The distribution of the loans into three classes:
raw_lab['Loan_response'].value_counts()

## 3. let's start cleanning the data 

##    a. EDA on the features, research background
    
##    b. creating dummie features from selected categorical featuresn(e.g. 'term' variable)
    
##    c. Discarding un-informative features (e.g. 'id' feature)
    
##    d. Dealing with missing data
        
#### a. EDA - Hypothesis is that some of the 'current' class attributes might look more like 
#'fully paid' and some more like 'Default'. So we can Put emphasis on loans that we know 
# that their final outcome, and discard for now the current class in our exploration


''''''

##    c. Discarding un-informative features (e.g. 'id' feature)


## id columns: Every row has a unique loan and is by a unique member. so we can drop these columns (even though there might be a change in id once the loan status changes?)
print('ids are unique:',len(raw_lab['id'].unique()), len(raw_lab['member_id'].unique()))

# interst rates column: are in strings, we should 'numerize' them 
raw_lab['int_rate']=raw_lab['int_rate'].apply(lambda x: x.rstrip('%'))
raw_lab['int_rate']=raw_lab['int_rate'].astype(float)

# Changing the grades categories to ranked numerals: grade - (1-7) and sub_grade (1-35)
grade_let=raw_lab['grade'].value_counts().index.sort_values()
subgrade_let=raw_lab['sub_grade'].value_counts().index.sort_values()

def lettonum(letters):
    d={}    
    for ii,ob in enumerate(letters):
        d[ob]=ii+1
    return(d)

raw_lab['grade']=raw_lab['grade'].replace(lettonum(grade_let))
raw_lab['sub_grade']=raw_lab['sub_grade'].replace(lettonum(subgrade_let))

## If time allows, could feature engineer titles with NLP ('manager', 'principal', 'attorny', 'Doctor', 'Physician', 'VP') 
## in one group of high earners, and the rest in other earners.
## for now I will drop it
print('# of employment titles:',len(raw_lab['emp_title'].unique()))

## employement length feature - turnning into numerical (note >10 is converted to 15)
emp_num=raw_lab['emp_length'].value_counts().index.sort_values()
print(emp_num)
emp_dic={'1 year':1, '10+ years':15, '2 years':2, '3 years':3, '4 years':4, '5 years':5,
       '6 years':6, '7 years':7, '8 years':8, '9 years':9, '< 1 year':0.5, 'n/a':0}
raw_lab['emp_length']=raw_lab['emp_length'].replace(emp_dic)

## ANY has just one entry in 'home_ownership'
one_dic={'ANY':'MORTGAGE'}
raw_lab['home_ownership']=raw_lab['home_ownership'].replace(one_dic)

## merging wedding with other because of just 7 instances
len(raw_lab[raw_lab['purpose']=='wedding'])
len(raw_lab[(raw_lab['Loan_response']==2)&(raw_lab['purpose']=='wedding')])
wed_dic={'wedding':'other'}
raw_lab['purpose']=raw_lab['purpose'].replace(wed_dic)

## Payment plan is when a loan is already charged-off (should confirm), so it's possibly target-leaked
## should remove feature
raw_lab['pymnt_plan'][raw_lab['Loan_response']==2].value_counts()
raw_lab['pymnt_plan'][raw_lab['Loan_response']==0].value_counts()
raw_lab['pymnt_plan'][raw_lab['Loan_response']==1].value_counts()

## For NLP engineering, should go into if there is time. for now - drop
len(raw_lab['title'].unique())

# ## Question - how to replace the nulls in 'mths_since_last_record' ? (same goes to 'mths_since_last_delinq' )
## null number too big (can not erase rows and 'mode', 'median' replacement might skew)
## 'mths_since_last_record' has:
## the same distribution in all classes (μ,σ,hist)
## the same ratio between nulls and numericals
## seems that there is no "information" in the feature
## but there is 2 apperant distributions: up to 24 months 'rec', and older 'old'
# turn: mths_since_last_record into 3 classes to later be dummie variables

raw_lab['mths_since_last_record.eng']=raw_lab['mths_since_last_record'].apply(lambda x: 'rec' if (x<25) else ('old' if x>24 else x))
raw_lab['mths_since_last_record.eng']=raw_lab['mths_since_last_record.eng'].replace(np.nan, 'NaN', regex=True)
# raw_lab['mths_since_last_record.eng'].value_counts()

#same with 'mths_since_last_delinq'
raw_lab['mths_since_last_delinq.eng']=raw_lab['mths_since_last_delinq'].apply(lambda x: 'rec' if (x<25)&(x>0) else ('old' if x>24 else x))
raw_lab['mths_since_last_delinq.eng']=raw_lab['mths_since_last_delinq.eng'].replace(np.nan, 'NaN', regex=True)
raw_lab['mths_since_last_delinq.eng'].replace(0,'NaN',inplace=True)
raw_lab['mths_since_last_delinq.eng'].value_counts()

# 'revol_util' are % in strings, and have nulls we should 'numerize', and replace nulls with 'Mode' = 0
raw_lab['revol_util']=raw_lab['revol_util'].replace(np.nan,'0', regex=True)
raw_lab['revol_util']=raw_lab['revol_util'].apply(lambda x: x.rstrip('%'))
raw_lab['revol_util']=raw_lab['revol_util'].astype(float)

## 'next_pymnt_d' - replace one instance of a date with one of the other dates (there are only 2)
## can dumify, or discard
raw_lab['next_pymnt_d']=raw_lab['next_pymnt_d'].replace('Oct-14', 'Nov-14', regex=True)

# ## Question - how to replace the nulls in 'mths_since_last_major_derog.eng' ? (same goes to mths_since_last_record','mths_since_last_delinq' )
## null number too big (can not erase rows and 'mode', 'median' replacement might skew)
## # seems that class default have more recent major derogetory than class fully paid
# Current and fully paid are similar  
# the classes have:
## the same distribution in all classes (μ,σ,hist)
## the same ratio between nulls and numericals
## for now i divided it as up to 24 months 'rec', and older 'old'
# turn: mths_since_last_record into 3 classes to later be dummie variables

raw_lab['mths_since_last_major_derog.eng']=raw_lab['mths_since_last_major_derog'].apply(lambda x: 'rec' if (x<25) else ('old' if x>24 else x))
raw_lab['mths_since_last_major_derog.eng']=raw_lab['mths_since_last_major_derog.eng'].replace(np.nan, 'NaN', regex=True)
raw_lab['mths_since_last_major_derog.eng'].value_counts()

## 'earliest_cr_line' should be engineered with 'issue_d' to 
## create a new feature - the number of years a borrower has a credit line:

monyr=raw_lab['earliest_cr_line'].str.split('-',expand=True)
monyr=monyr.iloc[:,1].astype(float)
monyr=monyr.apply(lambda x: (2000+x) if x<15 else (1900+x))
monyr2=raw_lab['issue_d'].str.split('-',expand=True)
monyr2=monyr2.iloc[:,1].astype(float)
monyr2=monyr2.apply(lambda x: (2000+x) if x<15 else (1900+x))
credyrs=monyr2-monyr   
raw_lab['credit_yrs']=credyrs


''' final steps of cleanning'''
'''-------------------------'''

# let's save the following major changes in a new df - clean_df
clean_df=raw_lab.copy()

# change the condition to TRUE to run the following lines

## columns "to dummiefy"    
# turn to TRUE to run (only after collecting all the columns that need it)
if 1==1:
    dums=['home_ownership','term','is_inc_v','purpose','mths_since_last_record.eng',
          'mths_since_last_major_derog.eng','mths_since_last_delinq.eng','initial_list_status','next_pymnt_d']
    clean_df=pd.get_dummies(raw_lab,columns=dums,drop_first=False)

## the following columns are floats without missing data, (whether originally or ammended as such)    
col_float=['loan_amnt', 'funded_amnt', 'funded_amnt_inv','int_rate', 'installment',
         'grade','sub_grade','emp_length','annual_inc','delinq_2yrs','dti',
         'inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util',
         'total_acc','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv',
         'total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries',
         'collection_recovery_fee','last_pymnt_amnt','collections_12_mths_ex_med']
    
## Columns to drop:
if 1==1:
    col_drop=['id','member_id','emp_title','loan_status','url','desc','pymnt_plan',
              'title','mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq']
    clean_df.drop(col_drop, inplace=True, axis=1)
    
## confirming the list of cols to keep with 'col_rel'
print('columns not in "clean_df", not in "col_drop" and not in "dums":')
print('-------------------------------------------')
for col in col_rel:
    if col not in clean_df.columns:
        if col not in col_drop:
            if col not in dums:
                print(col)
                
## confirming the list of cols to keep with 'col_rel'
#print('columns in "clean_df" but not in "col_rel":')
print('-------------------------------------------')
col_drop2=[]
for col in clean_df:
    if col not in col_rel:
        col_drop2.append(col)         
        
# cols to finaly drop:
col_drop3=col_drop2[0:17]+col_drop2[-3:-1] 

## Columns to drop:
if 1==1:
    clean_df.drop(col_drop3, inplace=True, axis=1)
    
## this our final clean df:
clean_df.info()

       
        

## by looking at clean_df.info() we can see 'object' and 'nulls', we can see the remainning problematic columns
col_prob=['last_pymnt_d','last_credit_pull_d','earliest_cr_line','issue_d','zip_code','addr_state']

## let's drop them for now just to get an initial df for modeling
clean_df.drop(col_prob, inplace=True, axis=1)

'''columns unsure how to use (should not stay in dataset as is)
---------------------------------------------------------------
dates: 'issue_d','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d'
location: 'zip_code','addr_state'
credit specific: 'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt','collections_12_mths_ex_med' '''

col_cred_spec=['total_rec_late_fee','recoveries','collection_recovery_fee',
               'last_pymnt_amnt','collections_12_mths_ex_med']
# all of these are in clean_df but would need to attend to 

''' Now we have the most basic clean df that allows us to get initial modeling '''    

    

''' #################################
    draft, don't run from this point 
    #################################'''


'''MODELING '''   



'''importing models '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

'''importing metrics '''
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


    


  