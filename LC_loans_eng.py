#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:54:46 2019

@author: eran
"""

''' Before running the script, change the following statement to 1, if to run with plots and some printing'''

run_with_plots=0
 

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


directory_path='/Users/eran/Galvanize_more_repositories/LendingClub_Loans'

# 1. Let's read in the data 
raw=pd.read_csv(os.path.join(directory_path,'LoanStats3c.csv'))
raw.head(2)

# to plot
if run_with_plots!=0:
    
    plt.subplots(figsize=(16,3))
    sns.heatmap(raw.isnull(),cbar=False)
    iii=plt.title('nulls in raw dataset')
    
'''Features to drop/keep
After LC online research, and using the dictionary given. the columns (fields) that should stay and ones 
that should be removed are given in the following summary I created:'''

col_rel=pd.read_csv('/Users/eran/Galvanize_more_repositories/LendingClub_Loans/col_relevant.csv',)
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

if run_with_plots!=0:
    plt.subplots(figsize=(6,2))
monyr=raw_lab['earliest_cr_line'].str.split('-',expand=True)
monyr=monyr.iloc[:,1].astype(float)
monyr=monyr.apply(lambda x: (2000+x) if x<15 else (1900+x))
monyr2=raw_lab['issue_d'].str.split('-',expand=True)
monyr2=monyr2.iloc[:,1].astype(float)
monyr2=monyr2.apply(lambda x: (2000+x) if x<15 else (1900+x))
credyrs=monyr2-monyr   
raw_lab['credit_yrs.eng']=credyrs

if run_with_plots!=0:
    raw_lab['credit_yrs.eng'].hist(bins=50)
    print('Borowers years of credit histogram')


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
    #clean_df=pd.concat([clean_df, dums], axis=1)

## the following columns are floats without missing data, (whether originally or ammended as such)    
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
              'title','mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq',
              'earliest_cr_line','next_pymnt_d_Dec-14','next_pymnt_d_Nov-14']
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
col_drop3=col_drop2[0:17]
col_drop3_leave_leak=col_drop3.copy()
col_drop3_leave_leak.remove('out_prncp')
len(col_drop3_leave_leak)

## create df with a "leake" column, as a control df:

leaked_df=clean_df.copy()
leaked_df.drop(col_drop3_leave_leak, inplace=True, axis=1)
#
# Columns to drop:
clean_df.drop(col_drop3, inplace=True, axis=1)
    
## confirming the difference between leaked and clean df:
print('confirming the difference between "leaked" and "clean" dfs:')
print('-------------------------------------------')
for col in leaked_df.columns:
    if col not in clean_df.columns:
        print(col)         
        
## this our final clean df:
if run_with_plots!=0:
    clean_df.info()

      
'''columns that were removed from dataset due to different reasons (target/temporal leak, uninformative, data type etc..:
---------------------------------------------------------------
dates: 'issue_d','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d'
location: 'zip_code','addr_state'
credit specific: 'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt','collections_12_mths_ex_med' '''


''' Now we have the most basic clean df that allows us to get initial modeling '''    

    

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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

'''import data-balancing package'''
from imblearn.over_sampling import SMOTE


'''Preparing train and test for "deafault" and "fully paid" classes'''

## data set of only classes 0 and 1:
def split_01(df):
    ds_01=df[df['Loan_response']!=2]
    ## clean_df[clean_df['Loan_response']==1].shape = (1838, 63)
    print('shape of dataset',ds_01.shape)
    return(ds_01)
## data set of classes 2:
def split_2(df):
    ds_2=df[df['Loan_response']==2]
    ## clean_df[clean_df['Loan_response']==1].shape = (1838, 63)
    print('shape of dataset',ds_2.shape)
    return(ds_2)
    

# saving the valid "current" data set:
ds2v=split_2(clean_df)


''' baseline modeling'''


def run_model(df,balance=False,onetwo=True,show_report=False):
    
    # subsetting the dataset for only classes 1 and 2
    if onetwo==True:
        ds01=split_01(df)
        
    # remember to input ds01v for valid dataset or ds01f for leaked dataset:
    #ds01=df
    # splitiing train, test
    X01_train, X01_test, y01_train, y01_test = train_test_split(ds01.drop('Loan_response',1),
                                                                ds01['Loan_response'], test_size=0.30, random_state=42) 
    column_names=X01_train.columns
    
    if balance==True:    
        smt=SMOTE(random_state=42)
        X01_train,y01_train=smt.fit_sample(X01_train,y01_train)
        print('balancing...:')
        
    
    print('shapes of train and test:') 
    print(X01_train.shape, X01_test.shape, y01_train.shape, y01_test.shape)
    print('---------------') 
    
    '''Instantiate a basic model (without tunning parameters)'''
    
    int_params={'n_estimators':500,'min_samples_split':3,'max_features':20,'max_depth':5}
    
    rf_int=RandomForestClassifier(random_state=42,n_estimators=int_params['n_estimators'],
                                 min_samples_split=int_params['min_samples_split'],
                                 max_features=int_params['max_features'],
                                 max_depth=int_params['max_depth'])
    
    rf_int.fit(X01_train,y01_train)
    predy=rf_int.predict(X01_test)
    predprob=rf_int.predict_proba(X01_test)
    cm=confusion_matrix(predy,y01_test)
    # fpr, tpr, thresholds = roc_curve(ytest, predprob[:,1])
    # precision, recall, threshol=precision_recall_curve(ytest, predprob[:,1])
    
    '''results for dataset'''
    
    print('confusion matrix:')
    print(cm)
    #print(classification_report(y01_test,predy))
    
    most=rf_int.feature_importances_[rf_int.feature_importances_>0.02]
    collnum=np.where(rf_int.feature_importances_>0.02)
    colll=column_names[collnum]  
    plt.subplots(figsize=(8,3))
    plt.barh(colll,most)
    j=plt.title('Features Importance')
    classif=classification_report(y01_test,predy)
    if show_report==True:
        print(classif)
        
        

print('results for clean basic dataset:')
run_model(clean_df)

print('results for target-leaked dataset:')
run_model(leaked_df)

print('results for clean basic and BALANCED dataset:')
run_model(clean_df,balance=True,show_report=True)

'''the following run of the engineered dataset was moved to the the end of the script:'''
# print('results for clean basic and BALANCED dataset:')
# run_model(eng_df,balance=True,show_report=True)

'''FEATURE ENGINEERING'''

#creating a new df for the final engineered dataset
eng_df=clean_df.copy()

# relative frequency function,to plot differences between 2 classes
def rel_freq(df,feat):
    efes=df[feat][df['Loan_response']==0]
    ehad=df[feat][df['Loan_response']==1]
    count0, division0 = np.histogram(efes)
    count1, division1 = np.histogram(ehad,bins=division0)
    prob0=count0/sum(count0)
    prob1=count1/sum(count1)
    plt.subplots(figsize=(6,2))
    plt.plot(division0[0:10],prob0,label="Fully Paid")
    plt.legend(loc='best')
    plt.plot(division0[0:10],prob1,label="Default")
    plt.legend(loc='best')
    plt.title('Relative frequency of'+' "'+feat+'" '+'of borrowers')

'''FEATURE 1'''
'''---------'''

'''mths_since_last_major_derog'''

print('printing a summary of information on the feature:')
coll='mths_since_last_major_derog'
print('"mths_since_last_major_derog"')
print('-----------------------')
for ii in [0,1,2]:
    raw_lab[coll][raw_lab['Loan_response']==ii].value_counts()
    tt=sum(raw_lab[coll][raw_lab['Loan_response']==ii]>0)
    mm=len(raw_lab[coll][raw_lab['Loan_response']==ii])
    print('class:',ii)
    print('e.g.:',raw_lab[coll][raw_lab['Loan_response']==ii].head(1))
    print('# nulls =',sum(raw_lab[coll][raw_lab['Loan_response']==ii].isnull()))
    print('# portion of nulls =',sum(raw_lab[coll][raw_lab['Loan_response']==ii].isnull())/len(raw_lab[coll][raw_lab['Loan_response']==ii]))
    print(raw_lab[coll][raw_lab['Loan_response']==ii].describe()[0:3])
    print(tt)
    print(mm)
    print('class',ii,': portion of values> 0 in the class:',np.round(tt/mm,3))
    print('-----------------------')
    
# seems that class default have more recent major derogetory than class fully paid
# Current and fully paid are similar 
efes=raw_lab[coll][raw_lab['Loan_response']==0]
ehad=raw_lab[coll][raw_lab['Loan_response']==1]

if run_with_plots!=0:  
    count0, division0 = np.histogram(efes[efes.notnull()])
    count1, division1 = np.histogram(ehad[ehad.notnull()],bins=division0)
    prob0=count0/sum(count0)
    prob1=count1/sum(count1)
    plt.subplots(figsize=(6,2))
    plt.plot(division0[0:10],prob0,label="Fully Paid")
    plt.legend(loc='best')
    plt.plot(division0[0:10],prob1,label="Default")
    plt.legend(loc='best')
    j=plt.title('Relative frequency of "mths_since_last_major_derog" of borrowers')

'''The variable seems relevant up to 60 months mark. it looks like different distributions 
where the cutoff is till 24 months and 24 to 60 months. these should be 
the columns: 'rec', 'old','irrelevant'( >60 including NaNs)'''

# engineering feature:
eng_df['mths_since_last_major_derog.eng']=raw_lab['mths_since_last_major_derog'].apply(lambda x: 'rec' if (x<25) else ('old' if (x>24) & (x<61) else 'NaN'))
eng_df['mths_since_last_major_derog.eng']=eng_df['mths_since_last_major_derog.eng'].replace(np.nan, 'NaN', regex=True)
eng_df['mths_since_last_major_derog.eng'].value_counts()

# creating dummie for the feature:
eng_df=pd.get_dummies(eng_df,columns=['mths_since_last_major_derog.eng'])
# removing NaN dummie :
eng_df.drop(['mths_since_last_major_derog.eng_NaN'], inplace=True, axis=1)


'''FEATURE 2'''
'''---------'''

'''Credit_yrs.eng'''

# Credit_yrs is engineered previously from 'issue_d' substracted by 'first_credit_line'
# By looking at the graphs we can see a slight change, where defaulting 
# borrowers have slightly less yrs of credit as expected

if run_with_plots!=0:
    rel_freq(eng_df,'credit_yrs.eng')

# let's take the log of these years, it makes sense because pressumably 
#the difference between 1 and 5 is much greater thasn 15-20
eng_df['log_credit_yrs.eng']=np.log10(eng_df['credit_yrs.eng'])

if run_with_plots!=0:
    rel_freq(eng_df,'log_credit_yrs.eng')

# removing the original column :
eng_df.drop(['credit_yrs.eng'], inplace=True, axis=1)

'''FEATURE 3'''
'''---------'''

'''annual_inc'''

## income is highly skewed with several outliers that are skewing the data. 
## again we'll use log-transformation:

if run_with_plots!=0:
    plt.subplots(figsize=(16,3))
    plt.subplot(1,1,1)
    sns.boxplot('annual_inc','Loan_response',data=clean_df,orient="h")
    sns.boxplot('annual_inc','Loan_response',data=clean_df,orient="h")

eng_df['annual_inc_log.eng']=np.log10(eng_df['annual_inc'])

## By looking at the graphs we can see a slight shift, where defaulting 
# borrowers have slightly less income as expected
if run_with_plots!=0:
    rel_freq(eng_df,'annual_inc_log.eng')

# removing the original column :
eng_df.drop(['annual_inc'], inplace=True, axis=1)


'''FEATURE 4'''
'''---------'''
''' State
How shall we use location data?

zipcode is probably informative. it has just 3 digits (hence with low resoulution). Ideally
we should engineer with longitude latitude converter or/and to enrich with census (translate zip 
to socio-economic level of neighborhood) this we'll leave for following work on project. Perhaps we can use 
state? it is less informative than zip, but we can quickly use a proxy such as median income per 
state (in 2014) which is publicly available:
https://www.reinisfischer.com/median-household-income-us-state-2014'''

state_inc=pd.read_csv('/Users/eran/Galvanize_more_repositories/LendingClub_Loans/state_income.csv',)

## retrieving the state code and the median state income
sor=state_inc[['Unnamed: 2','Unnamed: 3']]
## retrieving income and state code features from dataset
h=raw[['addr_state','annual_inc']]
dicc= dict(zip(sor['Unnamed: 2'],sor['Unnamed: 3']))

## engineering log ratio of individual income VS state median income:
eng_df['log_annual_inc_state_med.eng']=h['addr_state']
eng_df['log_annual_inc_state_med.eng']=eng_df['log_annual_inc_state_med.eng'].replace(dicc)
eng_df['log_annual_inc_state_med.eng']=np.log10(h['annual_inc']/eng_df['log_annual_inc_state_med.eng'])

# we can see that the feature seems to not differentiate better than regular log income
if run_with_plots!=0:
    rel_freq(eng_df,'log_annual_inc_state_med.eng')
    
'''FEATURE 5'''
'''---------'''

'''loan ammount / annual income '''
## let's maybe use loan ammount devided by the annual income, this ratio 
## should give us a stronger signal than anyone of the features alone   

eng_df['loan_amnt_to_inc.eng']=clean_df['loan_amnt']/clean_df['annual_inc']

if run_with_plots!=0:
    rel_freq(eng_df,'loan_amnt_to_inc.eng')

## let's keep the indipendant loan ammount feature as it might be informative by itself

'''FEATURE 6'''
'''---------'''  

''' Principal '''  
## we have data on the installment which includes the int_rate too. this means that the risk projected
## by lending club is already projected in the instalment. to create an additional  indipendant feature  
## we can use the principal which we don't have (the ammount indipendantly of the interest)
## but we can easily calculate it and add it
eng_df['prncp.eng']=eng_df['installment']-(eng_df['installment']*eng_df['int_rate']/100)

if run_with_plots!=0:
    rel_freq(eng_df,'prncp.eng')

## seems that there is again slight difference in the principal

'''results for model ran on engineered data:
    ---------------------------------------- '''

print('results for ENGINEERED and BALANCED dataset:')
run_model(eng_df,balance=True,show_report=True)




''' #################################
    draft, don't run from this point 
    #################################'''


'''#####################'''
# to compare dfs: 
for col in col_prob:
    if col not in col_drop3:
        print(col)



if 1==0:
    '''##############################################'''
    '''CURRENT'''
    
    ''' Let's prepare the "current" data (subsetting and train and test for later use)'''
    ## class 2 has ~ 150,000 obj, so let's subset a ~10,000 (i.e. "small") and leave the rest ~140,000 for later:
    X2_small, X2_big, y2_small, y2_big = train_test_split(ds2v.drop('Loan_response',1),
                                                            ds2v['Loan_response'], test_size=0.93, random_state=42)