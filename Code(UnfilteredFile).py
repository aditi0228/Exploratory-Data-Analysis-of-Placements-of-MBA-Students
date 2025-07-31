#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data=pd.read_csv("C:/Users/yash1/OneDrive/Desktop/project/Placement_Data_Full_Class.csv")


# In[3]:


data  #dataframe of whole class


# In[4]:


data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.dtypes


# In[17]:


data.head()


# In[18]:


df=data[data.status=="Placed"]


# In[19]:


df  #dataframe of  palaced students


# In[20]:


df.isnull().sum()


# In[21]:


df2=data[data.status!="Placed"]


# In[22]:


df2


# In[23]:


df2=df2[df2.salary.notnull()]


# In[24]:


df2


# In[25]:


df2=data[data.status!="Placed"]


# In[26]:


df2.dtypes


# In[27]:


df2.salary=df2.salary.replace(np.nan,0)


# In[28]:


df2 #dataframe of not palaced students


# In[37]:


data.fillna(0,inplace= True)


# In[38]:


df.describe()


# In[39]:


df2.describe()


# In[40]:


data['Degree stream'].unique().shape


# In[41]:


data['specialisation'].unique().shape


# In[42]:


data['Gender'].unique()


# In[27]:


data.head()


# In[28]:


data


# In[29]:


data.nlargest(10,'12th %')['Gender']


# In[30]:


plt.bar(np.arange(2)-0.3,[data['Gender'].value_counts()[0],data['Gender'].value_counts()[1]], width=0.3)
plt.bar(np.arange(2),[df['Gender'].value_counts()[0],df['Gender'].value_counts()[1]],width=0.3)
plt.bar(np.arange(2)+0.3,[df2['Gender'].value_counts()[0],df2['Gender'].value_counts()[1]],width=0.3)


# In[36]:


data['Gender'].value_counts()[0]


# #  Univariate Analysis

# In[30]:



values1 = [139, 76]
values2 = [100, 48]


# Set the positions of the bars on the x-axis
ind = np.arange(len(categories))
width = 0.40

# Create the figure and the axes
fig, ax = plt.subplots()

# Plot the bars
rects1 = ax.bar(ind - width/2, values1, width, label='Total')
rects2 = ax.bar(ind + width/2, values2, width, label='Placed')


# Add labels, title, and legend
ax.set_xlabel('Gender')
ax.set_ylabel('Values')
ax.set_title('Placement_status vs Gender')
ax.set_xticks([-0.2,0.2,0.8,1.2])
ax.set_xticklabels(['Males','M_Placed','Females','F_Placed'])
ax.legend()
indices=[-0.2,0.2,0.8,1.2]

values = [total_males, males_placed_count, total_females, females_placed_count]
for index, value in zip(indices,values):
    plt.text(index, value, str(value), ha='center', va='bottom')
# Show the plot
plt.show()


# In[57]:


plt.pie([100,39],labels=["Placed",""],autopct='%1.1f%%')
plt.title("Proportion of Male Students_Placed")


# In[69]:


plt.pie([48,28],labels=["Placed",""],autopct='%1.1f%%')
plt.title("Proportion of Female Students_Placed")


# In[127]:


(100/76)-(48/76)


# In[ ]:


#Proportion of students who got placed with Mkt&Fin


# In[297]:


df[df['specialisation']=='Mkt&Fin']['Sno'].count()


# In[298]:


df[df['specialisation']=='Mkt&HR']['Sno'].count()


# In[299]:


plt.pie([95,25],labels=["Placed",""],autopct='%1.1f%%')
plt.title("Placement rate of Finance Students")


# In[301]:


plt.pie([53,42],labels=["Placed",""],autopct='%1.1f%%')
plt.title("Placement rate of HR Students")


# In[ ]:


=95/53-55/


# In[ ]:


# males females ratio in data


# In[99]:


fig,ax= plt.subplots()
plt.pie(data['Gender'].value_counts(),labels=["Male"+"("+ str(total_males)+")","Female"+"("+ str(total_females)+")"],autopct='%1.1f%%')

plt.show()


# In[ ]:


# Proportion of males/Females in ssc and hsc board


# In[38]:


total_males = data[data['SSC Board'] == 'Central'].shape[0]
total_females = data[data['SSC Board'] == 'Others'].shape[0]

fig,ax= plt.subplots()
plt.pie(sorted(data['SSC Board'].value_counts(),reverse=True),labels=["CENTRAL"+"("+ str(total_males)+")","OTHERS"+"("+ str(total_females)+")"],autopct='%1.1f%%')

plt.show()


# In[109]:


total_males = data[data['HSC Board'] == 'Central'].shape[0]
total_females = data[data['HSC Board'] == 'Others'].shape[0]

fig,ax= plt.subplots()
plt.pie(sorted(data['HSC Board'].value_counts()),labels=["CENTRAL"+"("+ str(total_males)+")","OTHERS"+"("+ str(total_females)+")"],autopct='%1.1f%%')

plt.show()


# In[39]:


#Proportion of students in various stream


# In[42]:


total_sci = data[data['12th Stream'] == 'Science'].shape[0]
total_comm = data[data['12th Stream'] == 'Commerce'].shape[0]
total_arts = data[data['12th Stream'] == 'Arts'].shape[0]

plt.pie((data['12th Stream'].value_counts()).sort_index(),labels=["Arts"+"("+ str(total_arts)+")","Commerce"+"("+ str(total_comm)+")","Science"+"("+ str(total_sci)+")"],autopct='%1.1f%%')


# In[43]:


# Proportion of the students in specializtion


# In[50]:


total_sci = data[data['Degree stream'] == 'Sci&Tech'].shape[0]
total_comm = data[data['Degree stream'] == 'Comm&Mgmt'].shape[0]
total_arts = data[data['Degree stream'] == 'Others'].shape[0]

plt.pie((data['Degree stream'].value_counts()).sort_index(),labels=["Comm&Mgmt"+"("+ str(total_comm)+")","Others"+"("+ str(total_arts)+")","Sci&Tech"+"("+ str(total_sci)+")"],autopct='%1.1f%%')


# In[51]:


#Distribution of marks in class 10th


# In[114]:


sns.histplot(data['10th %'], bins=10, kde=True, color='black', edgecolor='black')


# In[ ]:


#Distribution of marks in class 12th


# In[115]:


sns.histplot(data['12th %'], bins=10, kde=True, color='blue', edgecolor='black')


# In[ ]:


#Distribution of marks in Degree


# In[52]:


sns.histplot(data['Degree %'], bins=10, kde=True, color='blue', edgecolor='black')


# In[ ]:


#Distribution of marks in Mba


# In[54]:


sns.kdeplot(data['Mba %'], shade=True, color='blue')


# In[154]:


data['10th %'].mean()+data['10th %'].std()


# In[156]:


num_points= np.sum(((data['10th %']) >=56.47618995060574 ) & (data['10th %'] <=78.13060074706866 ))
total_points = len(data['10th %'])
print(num_points,total_points)
probability = num_points / total_points

probability


# In[162]:


sns.boxplot(x='SSC Board', y="10th %", data=data, palette="Set3")


# In[164]:


sns.boxplot(x='HSC Board', y="12th %", data=data, palette="Set3")


# In[165]:


sns.boxplot(x='12th Stream', y="12th %", data=data, palette="Set3")


# In[55]:


sns.boxplot(x='specialisation', y="Mba %", data=data, palette="Set3")


# In[80]:


sns.boxplot( y="salary", data=df,  color='blue',width=0.6,         # Width of the boxes
            linewidth=2,     # Width of the whiskers
            fliersize=2,       # Size of the outlier points
            notch=True,        # Notch around median
            showmeans=True)

plt.title('Box Plot for salary')
plt.xlabel('')
plt.ylabel('salary')
plt.show()


# In[43]:


sns.boxplot(x='specialisation', y='salary', data=df, palette="Set3")


# In[13]:


data['10th %'].describe()


# In[ ]:


data[data['10th %'].describe()


# In[81]:


# Salary Statistics


# In[115]:


print("Average Package =",np.ceil(df['salary'].count()))
print("Median Package =",np.ceil(df['salary'].median()))
print("Maximum Package =",np.ceil(df['salary'].max()))
print("Minimum Package =",np.ceil(df['salary'].min()))
print("Maximum Package in Mkt&Fin =",np.ceil(df[df['specialisation']=='Mkt&Fin']['salary'].max()))
print("Maximum Package In Mkt&HR =",np.ceil(df[df['specialisation']=='Mkt&HR']['salary'].max()))


# In[ ]:


#distribution of salary


# In[96]:


sns.histplot(df['salary'], bins=10, kde=True, color='blue', edgecolor='black')


# In[97]:


sns.histplot(df[df['specialisation']=='Mkt&Fin']['salary'], bins=10, kde=True, color='blue', edgecolor='black')


# In[108]:


df[df['specialisation']=='Mkt&Fin']['salary'].describe()


# In[ ]:


This indicates that there are more low-income earners, and a few high-income earners contribute to the longer tail on the right side.


# In[ ]:





# In[98]:


sns.histplot(df[df['specialisation']=='Mkt&HR']['salary'], bins=10, kde=True, color='blue', edgecolor='black')


# In[116]:


plt.figure(figsize=(8, 6))
sns.histplot(df[df['specialisation']=='Mkt&Fin']['salary'], bins=10,color='red', kde=True , edgecolor='black',label='Mkt&FIn')
sns.histplot(df[df['specialisation']=='Mkt&HR']['salary'], bins=10,color='blue', kde=True, edgecolor='black',label='Mkt&HR')
plt.legend()
plt.title('Histograms with Density Plot Depicting the Distribution of Salary')
plt.show()


# In[ ]:


The mean is typically greater than the median in a positively skewed distribution.
This indicates that there are more low-income earners, and a few high-income earners contribute to the longer tail on the right side.


# In[118]:


df[df['Gender']=='F']['salary'].max()


# In[114]:


(df[df['specialisation']=='Mkt&Fin']['salary']>450000).sum()


# In[295]:


#Placement Rate


# In[119]:


rate=148/215
round(rate*100,0)


# In[121]:


# Gender vs Stream vs specializtion


# In[139]:


gen=data[data['Gender']=='M']


# In[140]:


total_m1 = gen[gen['Degree stream'] == 'Sci&Tech'].shape[0]
total_m2 = gen[gen['Degree stream'] == 'Comm&Mgmt'].shape[0]
total_m3=gen[gen['Degree stream'] == 'Others'].shape[0]

fig,ax= plt.subplots()
plt.pie(gen['Degree stream'].value_counts().sort_index(),labels=["Comm&Mgmt"+"("+ str(total_m2)+")","Others"+"("+ str(total_m3)+")","Sci&Tech"+"("+ str(total_m1)+")"],autopct='%1.1f%%')

plt.show()


# In[ ]:


# Gender vs specializtion


# In[216]:


gen=data[data['Gender']=='F']


# In[217]:


total_m1 = gen[gen['Degree stream'] == 'Sci&Tech'].shape[0]
total_m2 = gen[gen['Degree stream'] == 'Comm&Mgmt'].shape[0]
total_m3=gen[gen['Degree stream'] == 'Others'].shape[0]

fig,ax= plt.subplots()
plt.pie(gen['Degree stream'].value_counts().sort_index(),labels=["Comm&Mgmt"+"("+ str(total_m2)+")","Others"+"("+ str(total_m3)+")","Sci&Tech"+"("+ str(total_m1)+")"],autopct='%1.1f%%')

plt.show()


# In[228]:


gen=data[data['Gender']=='F']


# In[229]:


total_f1 = gen[gen['specialisation'] == 'Mkt&Fin'].shape[0]
total_f2 = gen[gen['specialisation'] == 'Mkt&HR'].shape[0]
# total_m3=gen[gen['specialisation'] == 'Sci&Tech'].shape[0]

fig,ax= plt.subplots()
plt.pie(gen['specialisation'].value_counts().sort_index(),labels=["MKt&Fin"+"("+ str(total_f1)+")","Mkt&HR"+"("+ str(total_f2)+")"],autopct='%1.1f%%')


# In[230]:


gen=data[data['Gender']=='M']


# In[231]:


total_m1 = gen[gen['specialisation'] == 'Mkt&Fin'].shape[0]
total_m2 = gen[gen['specialisation'] == 'Mkt&HR'].shape[0]
# total_m3=gen[gen['specialisation'] == 'Sci&Tech'].shape[0]

fig,ax= plt.subplots()
plt.pie(gen['specialisation'].value_counts().sort_index(),labels=["MKt&Fin"+"("+ str(total_m1)+")","Mkt&HR"+"("+ str(total_m2)+")"],autopct='%1.1f%%')


# In[ ]:





# In[246]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data
Gender = ['Males', 'Females']
values1 = [total_m1,total_f1]
values2 = [total_m2, total_f2]

# Calculate the total values for each category
total_values1 = np.sum(values1)
total_values2 = np.sum(values2)

# Calculate the percentage values
percentage_values1 = [(value / total_values1) * 100 for value in values1]
percentage_values2 = [(value / total_values2) * 100 for value in values2]

# Plot
fig, ax = plt.subplots()

# Stacked bar plot
bars1=ax.bar(Gender, percentage_values1, label='Mkt&Fin', color='blue')
bars2=ax.bar(Gender, percentage_values2, bottom=percentage_values1, label='Mkt&HR', color='orange')


# Add labels and title
ax.set_xlabel('Gender')
ax.set_ylabel('Percentage(%)')
ax.set_title('Percentage of males and females in respective specialisations') # who opted for fin and percent of females who opted for fin
ax.set_ylim(0, 140,10)  # y-axis range
ax.set_yticks(np.arange(0, 141, 10))

for bar1, bar2 in zip(bars1, bars2):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.annotate(f'{height1:.1f}%', 
                xy=(bar1.get_x() + bar1.get_width() / 2, height1 / 2),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom')
    ax.annotate(f'{height2:.1f}%', 
                xy=(bar2.get_x() + bar2.get_width() / 2, height1 + height2 / 2),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom')
ax.legend()

# Display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# 12th Stream and percentage distribution


# In[ ]:


stream=data[data['12th Stream']=='Science']


# In[ ]:


plt.figure(figsize=(8, 6))
sns.histplot(df[df['stream']=='Science'], bins=10,color='red', kde=True , edgecolor='black',label='Mkt&FIn')
sns.histplot(df[df['specialisation']=='Mkt&HR']['salary'], bins=10,color='blue', kde=True, edgecolor='black',label='Mkt&HR')
plt.legend()
plt.title('Histograms with Density Plot Depicting the Distribution of Salary')
plt.show()


# In[175]:


plt.figure(figsize=(8, 6))
sns.histplot(data[data['specialisation']=='Mkt&Fin']['Mba %'], bins=10,color='red', kde=True , edgecolor='black',label='Mkt&FIn')
sns.histplot(data[data['specialisation']=='Mkt&HR']['Mba %'], bins=10,color='blue', kde=True, edgecolor='black',label='Mkt&HR')
plt.legend()
plt.title('Comparative Distribution of MBA% Across Two Specialization Streams: A Histogram and KDE Analysis')
plt.show()


# In[ ]:


a lot od students in finance scored above 60%, a little bit skewed


# In[176]:


data['Mba %'].describe()


# In[99]:


df


# ### Ques1: % of students who opted science in 12th but commerce in MBA
# ### Ques2:  Why a lot of them left central board is there any relation with 10th percent
# ### Ques3: Association of percent in class 12th and choice of Degree stream
# ### Ques4:Work Exp+Placed
# ### Ques5: Distribution of salary for work exp/ without work exp--->

# In[280]:


plt.figure(figsize=(8, 6))
sns.histplot(df[df['Work exp']=='No']['salary'], bins=10,color='red', kde=True , edgecolor='black',label='No')
sns.histplot(df[df['Work exp']=='Yes']['salary'], bins=10,color='blue', kde=True, edgecolor='black',label='Yes')
plt.legend()
plt.title('Histograms with Density Plot Depicting the Distribution of Salary')
plt.show()


# In[101]:


plt.figure(figsize=(8, 6))
sns.histplot(df[df['specialisation']=='Mkt&HR']['salary'], bins=10,color='red', kde=True , edgecolor='black',label='Mkt&HR')
sns.histplot(df[df['specialisation']=='Mkt&Fin']['salary'], bins=10,color='blue', kde=True, edgecolor='black',label='Mkt&Fin')
plt.legend()
plt.title('Histograms with Density Plot Depicting the Distribution of Salary')
plt.show()


# In[308]:


# Ques2:  Why a lot of them left central board is there any relation with 10th percent


# In[350]:


count_st= data[(data['SSC Board']=='Central')&(data['HSC Board']=='Others')]['Gender'].value_counts()
count_st


# In[288]:


per=count_st/ (data[data['SSC Board']=='Central']['Sno']).count()


# In[292]:


per*100


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'Reasoooon')


# In[302]:


sns.histplot(data['10th %'], bins=10,color='red' , edgecolor='black',label='No')


# In[23]:


new_df =data[(data['10th %']<=70)]
new_df[(new_df['SSC Board']=='Central')&(new_df['HSC Board']=='Others')]['Sno'].count()


# In[24]:


new_df.shape[0]


# In[18]:


data[(data['10th %']>67)&(data['SSC Board']=='Central')]['Sno'].count()


# In[348]:


data[(data['Gender']=='M')&(data['SSC Board']=='Central')]['Sno'].count()


# In[349]:


new_df =data[(data['Gender']=='M')]
new_df[(new_df['SSC Board']=='Central')&(new_df['HSC Board']=='Others')]['Sno'].count()


# In[346]:


13/42*100


# In[347]:


26/74*100


# In[353]:


ndf=data[(data['SSC Board']=='Central')&(data['HSC Board']=='Others')]
ndf[data['10th %']>67].shape


# In[354]:


#####Form a contingency table and check


# In[358]:


### Ques3: Association of percent in class 12th and choice of Degree stream
### Ques4:Work Exp+Placed

count_st= data[(data['12th Stream']=='Arts')&(data['Degree stream']=='Comm&Mgmt')]['Sno'].count()
count_st


# In[362]:


count_st= data[(data['12th %']<=data['12th %'].median())&(data['Degree stream']=='Comm&Mgmt')]['Sno'].count()
print(count_st)
(count_st/data[(data['12th Stream']=='Science')]['Sno'].count())*100


# In[363]:


data[(data['12th %']<=data['12th %'].median())].shape[0]


# In[378]:


count_st= data[(data['10th %']<=data['10th %'].median())&(data['12th Stream']=='Commerce')]['Sno'].count()
print(count_st)
(count_st/data[data['10th %']<=data['10th %'].median()]['Sno'].count())*100


# In[373]:


data[data['10th %']<=data['10th %'].median()-7]['Sno'].count()


# In[ ]:





# In[ ]:





# # Bivariate Analysis

# In[334]:


corr_coeff1 = data['Mba %'].corr(data['Degree %'], method='pearson')
corr_coeff1


# In[294]:


corr_coeff2 = data['12th %'].corr(data['Degree %'], method='pearson')
corr_coeff2


# In[251]:


corr_coeff3 = data['10th %'].corr(data['12th %'], method='pearson')
corr_coeff3


# ## 0.40 - 0.59: Moderate correlation

# In[252]:


sns.scatterplot(x= data['10th %'], y=data['12th %'], data=df, color='blue', alpha=0.7)


# In[260]:


correlation_matrix = data[ ['10th %',  '12th %',  'Degree %','Mba %','salary']].corr()
print(correlation_matrix)


# In[270]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')

# Add title
plt.title('Correlation Heatmap')
plt.show()


# In[268]:


sns.pairplot(data[ ['10th %',  '12th %',  'Degree %','Mba %','salary']], palette='Set2', diag_kind='kde', diag_kws=dict(shade=True),plot_kws=dict(edgecolor="b", alpha=0.7, linewidth=0.7))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Construction of contingency table for Independence of Attribute Test and R.B.D

# In[ ]:


#work_exp vs  placed


# In[177]:


c1=data[['Work exp','status']]
c2=data[['Work exp','status']]


# In[175]:


ww1=len((list(c1[c1['Work exp']=='Yes']['status'])))
ww2=len((list(c2[c2['Work exp']=='Yes']['status'])))


# In[178]:


ww1=len((list(c1[c1['Work exp']=='No']['status'])))
ww1


# In[ ]:





# In[ ]:





# In[128]:


c2.shape[0]


# In[133]:


columns = ['W_exp','Placed', 'Not Placed']
d = [
    ['Yes',ww1,ww2],
    ['No', c1.shape[0]-ww1,c2.shape[0]-ww2]
]
d1=pd.DataFrame(d, columns=columns)
d1


# In[155]:


c1=df[['Work exp','salary']]


ww1=(list(c1[c1['Work exp']=='Yes']['salary']))
ww2=(list(c1[c1['Work exp']=='No']['salary']))
max_len=min(len(ww1),len(ww2))
ww1=(list(c1[c1['Work exp']=='Yes']['salary']))+[np.nan] * (max_len - len(ww1)+1)

col1 = ['With_exp']
col2=['without_exp']
d1=pd.DataFrame(ww1, columns=col1)
d2=pd.DataFrame(ww2, columns=col2)
d1.fillna(0,inplace=True)
d1


# In[134]:


#Degree stream +Specialisation


# In[ ]:


#Degree vs placed


# In[ ]:


#specialisation vs placed


# In[164]:



from scipy.stats import shapiro
# Perform Shapiro-Wilk test
ww1=(list(c1[c1['Work exp']=='Yes']['salary']))
ww2=(list(c1[c1['Work exp']=='No']['salary']))
statistic, p_value = shapiro((data['Degree %']))

print("Shapiro-Wilk Test:")
print("Statistic:", statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Data is not normally distributed.")
else:
    print("Fail to reject the null hypothesis: Data is normally distributed.")


# In[ ]:





# In[165]:


from scipy.stats import kstest

# Perform Kolmogorov-Smirnov test
statistic, p_value = kstest(data['Degree %'], 'norm')

print("Kolmogorov-Smirnov Test:")
print("Statistic:", statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Data is not normally distributed.")
else:
    print("Fail to reject the null hypothesis: Data is normally distributed.")


# In[160]:


sns.kdeplot(ww2, shade=True, color='blue')


# In[ ]:


#work_exp+salary (out of placed students)


# In[ ]:





# In[136]:


#specialisation vs placed


# In[ ]:





# In[137]:


#specialisation vs salary


# In[ ]:





# In[ ]:


#rbd matrix


# In[ ]:





# In[93]:


data[(data['specialisation']=='Mkt&Fin') & (data['status']=='Placed') & (data['Mba %']>=(data['Mba %'].mean()))].shape[0]


# In[88]:


data['Mba %'].mean()


# In[166]:


## For RBD


# In[50]:


import pandas as pd
import numpy as np

# Assuming your dataset is stored in a DataFrame called 'df'

# Randomly shuffle the dataset
df_shuffled = data.sample(frac=1, random_state=42)

# Separate students with work experience and without work experience
students_with_exp = df_shuffled[df_shuffled['Work exp'] == 'Yes']
students_without_exp = df_shuffled[df_shuffled['Work exp'] == 'No']

# Randomly select 10 students with work experience
selected_students_with_exp = students_with_exp.sample(n=30, random_state=42)

# Randomly select 10 students without work experience
selected_students_without_exp = students_without_exp.sample(n=30, random_state=42)

# Concatenate the selected students into a single dataframe
selected_students = pd.concat([selected_students_with_exp, selected_students_without_exp])

# Display the selected students
print(selected_students)


# In[168]:


students_with_exp


# In[51]:


df_block2= students_without_exp[students_without_exp['Degree stream'] == 'Others']
df_block1= students_with_exp[students_with_exp['Degree stream'] == 'Others']
df_block4= students_without_exp[students_without_exp['Degree stream'] == 'Sci&Tech']
df_block3= students_with_exp[students_with_exp['Degree stream'] == 'Sci&Tech']
df_block6= students_without_exp[students_without_exp['Degree stream'] == 'Comm&Mgmt']
df_block5= students_with_exp[students_with_exp['Degree stream'] == 'Comm&Mgmt']


# In[188]:


df_block2.sample(n=4, random_state=2)


# In[189]:


df_block1


# In[190]:


df_block4.sample(n=4, random_state=2)


# In[191]:


df_block3.sample(n=4, random_state=2)


# In[194]:


df_block6.sample(n=4, random_state=4)


# In[125]:


list(df_block6.sample(n=4, random_state=2).status)


# In[54]:


df_block5[df_block5['status']=='Placed' & df_block5['Work_exp']].shape[0]


# In[55]:


39/45


# In[59]:


da=data[['Degree stream', 'Work exp','status']]
da2=da[da['status']=='Placed']


# In[60]:


da2


# In[61]:


proportions = da2.groupby(['Degree stream','Work exp']).count() /da.groupby(['Degree stream', 'Work exp']).count()
dfff=proportions

dfff


# In[58]:


da.groupby(['Degree stream', 'Work exp']).count()/da.groupby(['Degree stream'],).count()


# In[33]:


sns.kdeplot(df['salary'], shade=True, color='blue')


# In[57]:


df['salary'].describe()
b=df['salary'].copy()


# In[41]:


a=pd.DataFrame((np.array(df['salary'])-df['salary'].mean())/df['salary'].std())


# In[48]:


a.columns = ['salary']
a


# In[49]:


sns.kdeplot(a['salary'], shade=True, color='blue')


# In[59]:


sns.boxplot(y=a['salary'])


# In[58]:





# In[ ]:


iqr=
q3=


# In[74]:


df[(df['Work exp']=='Yes')& (df['specialisation']=='Mkt&Fin')]['salary'].describe()


# In[71]:


df[(df['Work exp']=='N')& (df['specialisation']=='Mkt&HR')]['salary'].describe()


# In[76]:


df[(df['Gender']=='F')& (df['specialisation']=='Mkt&Fin')]['salary'].describe()


# In[97]:


data[(data['Gender']=='F')& (data['specialisation']=='Mkt&HR')].describe()


# In[ ]:





# In[ ]:





# In[ ]:


# students who performed well in their academics have 100% placement rate


# In[229]:


toppers=data[ (data['10th %']> data['10th %'].median()) & (data['12th %']> data['12th %'].median() )&(data['Mba %']> data['Mba %'].median() )&(data['Degree %']> data['Degree %'].median() )]


# In[242]:


toppers


# In[231]:


((toppers['status']=='Placed').count()/toppers.shape[0])*100


# In[119]:


brilliant=data[ (data['10th %']>data['10th %'].median()) & (data['12th %']<data['12th %'].median() )&(data['Mba %']<data['Mba %'].median() )&(data['Degree %']< data['Degree %'].median()) & (data['Work exp']=='No')]


# In[120]:


brilliant


# ### LInear Reg Fit

# In[5]:


data[data['Work exp']=='Yes'].shape[0]


# # extra insights
# 

# In[55]:


data


# In[34]:


data['10th %'].describe()


# In[37]:


data['12th %'].describe()


# In[32]:


data[data['12th %']>90]


# In[35]:


data[data['Degree %']>90]


# In[20]:


# Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.
data[(data['10th %']<data['10th %'].quantile(0.25)-)]


# In[56]:


v='10th %'
iqr=data[v].quantile(0.75)-data[v].quantile(0.25)
q1=data[v].quantile(0.25)
q3=data[v].quantile(0.75)
data[(data[v]<q1-1.5*iqr) | (data[v]>q3+1.5*iqr) ].shape


# In[57]:


v='12th %'
iqr=data[v].quantile(0.75)-data[v].quantile(0.25)
q1=data[v].quantile(0.25)
q3=data[v].quantile(0.75)
data[(data[v]<q1-1.5*iqr) | (data[v]>q3+1.5*iqr) ]


# In[58]:


v='Mba %'
iqr=data[v].quantile(0.75)-data[v].quantile(0.25)
q1=data[v].quantile(0.25)
q3=data[v].quantile(0.75)
data[(data[v]<q1-1.5*iqr) | (data[v]>q3+1.5*iqr) ].shape


# In[59]:


v='Degree %'
iqr=data[v].quantile(0.75)-data[v].quantile(0.25)
q1=data[v].quantile(0.25)
q3=data[v].quantile(0.75)
data[(data[v]<q1-1.5*iqr) | (data[v]>q3+1.5*iqr) ]


# In[80]:


vs='12th %'
iqrs=data[vs].quantile(0.75)-data[vs].quantile(0.25)
qs1=data[vs].quantile(0.25)
qs3=data[vs].quantile(0.75)
data[(data[vs]<q1-1.5*iqr) | (data[vs]>q3+1.5*iqr) ]

v='Degree %'
iqr=data[v].quantile(0.75)-data[v].quantile(0.25)
q1=data[v].quantile(0.25)
q3=data[v].quantile(0.75)
data[(data[v]<q1-1.5*iqr) | (data[v]>q3+1.5*iqr) ]

dr1=data[((data[v]<q1-1.5*iqr) | (data[v]>q3+1.5*iqr))]
dr2=data[((data[vs]<qs1-1.5*iqrs) | (data[vs]>qs3+1.5*iqrs))]

outlierfree=data[((data[v]>q1-1.5*iqr) & (data[v]<q3+1.5*iqr)) &((data[vs]>qs1-1.5*iqrs) & (data[vs]<qs3+1.5*iqrs))]


# In[81]:


outlierfree.shape


# In[82]:


dr2


# In[83]:


drop=pd.concat([dr1, dr2])


# In[94]:


list()


# In[89]:


data.drop(drop[1:], axis=0)


# In[62]:


v='salary'
iqr=df[v].quantile(0.75)-df[v].quantile(0.25)
q1=df[v].quantile(0.25)
q3=df[v].quantile(0.75)


# In[67]:


outlierfree=df[(df['salary']>q1-1.5*iqr) & (df['salary']<q3+1.5*iqr)]


# In[68]:


outlierfree


# In[69]:


sns.kdeplot(outlierfree['salary'], shade=True, color='blue')


# In[80]:


outlierfree.to_csv('C:/Users/yash1/OneDrive/Desktop/project/outlierfree_data.csv', index=False)


# In[31]:


outlierfree


# In[79]:


from scipy.stats import kstest

# Perform Kolmogorov-Smirnov test
statistic, p_value = kstest(outlierfree['salary'], 'norm')

print("Kolmogorov-Smirnov Test:")
print("Statistic:", statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.
if p_value < alpha:
    print("Reject the null hypothesis: Data is not normally distributed.")
else:
    print("Fail to reject the null hypothesis: Data is normally distributed.")


# In[125]:


import scipy.stats as stats
stats.probplot(outlierfree['12th %'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.xlabel('Theoretical quantiles (MBA%)')
plt.ylabel('Ordered Values')
plt.grid(True)
plt.show()


# In[28]:



pop_std=df['salary'].std()


# In[29]:



sample = df['salary'].sample(n=30, replace=False)


# In[1]:


sample_mean=sample.mean()


# In[ ]:


import math
lower_bound= sample_mean-(1.96*(pop_std/(math.sqrt(30))))
upper_bound=sample_mean+(1.96*(pop_std/(math.sqrt(30))))
print(lower_bound,upper_bound)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Not placed reasons
# Palcement rate of females in different department
#relative risk rate=( placed| work exp)/(not placed |work exp)


# In[55]:


total_feamle=76
females_in_hr=data[(data['Gender']=='F') & (data['specialisation']=='Mkt&HR')].shape[0]
females_in_Fin=data[(data['Gender']=='F') & (data['specialisation']=='Mkt&Fin')].shape[0]
print(females_in_Fin,females_in_hr)


# In[56]:


total_feamle=76
females_in_hr=data[(data['Gender']=='F') & (data['specialisation']=='Mkt&HR')&(data['status']=="Placed")].shape[0]
females_in_Fin=data[(data['Gender']=='F') & (data['specialisation']=='Mkt&Fin')&(data['status']=="Placed")].shape[0]
print(females_in_Fin,females_in_hr)


# In[57]:


#rate of placement of female in Hr
rate= 28/37*100
rate


# In[58]:


rate= 20/39*100
rate


# In[48]:


df2[df2['Work exp']=='No' ]


# In[46]:


data[data['Work exp']=='No' ].count()


# In[47]:


prob_without_work_exp=84/141
prob_without_work_exp


# In[44]:


prob_with_work_exp=67/74
prob_with_work_exp


# In[73]:


df['salary']=(df['salary']-df['salary'].mean())/df['salary'].std()


# In[76]:


df[(df['salary']<3)|(df['salary']>-3)] 


# In[ ]:





# In[ ]:




