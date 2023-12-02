#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(r'diabetes.csv')

st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)



# HEADINGS
# st.title(':blue[Diabetes Checkup Project by] :orange[Shubham Rai]')
st.markdown("<h1 style='text-align: center; color: blue;'>Diabetes Checkup Project by <span style='color: orange;'>Shubham Rai</span></h1>", unsafe_allow_html=True)

#st.sidebar.title(':green[Filtering]')
#st.sidebar.header(':violet[Patient Data]')
st.sidebar.markdown("<h3 style='color: violet;'>Patient Data</h3>", unsafe_allow_html=True)

#st.subheader('Training Data Stats')
st.markdown("<span style='color: blue; font-size: 30px;'>Training Data Stats</span>", unsafe_allow_html=True)

st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
#st.subheader(':orange[Calculating Report for this Patient]')
st.markdown("<span style='color: green; font-size: 30px;'>Calculating Report for this Patient</span>", unsafe_allow_html=True)
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data.values)




# VISUALISATIONS
#st.title(':violet[Visualised Patient Report]')
#st.title("<span style='color: violet;'>Visualised Patient Report</span>", unsafe_allow_html=True)



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
#st.header(':blue[Pregnancy count Graph (Others vs Yours)]')
st.markdown("<span style='color: blue; font-size: 30px;'>Pregnancy count Graph (Others vs Yours)</span>", unsafe_allow_html=True)

fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
#st.header(':blue[Glucose Value Graph (Others vs Yours)]')
st.markdown("<span style='color: blue; font-size: 30px;'>Glucose Value Graph (Others vs Yours)</span>", unsafe_allow_html=True)

fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
#st.header(':blue[Blood Pressure Value Graph (Others vs Yours)]')
st.markdown("<span style='color: blue; font-size: 30px;'>Blood Pressure Value Graph (Others vs Yours)</span>", unsafe_allow_html=True)

fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
#st.header(':blue[Skin Thickness Value Graph (Others vs Yours)]')
st.markdown("<span style='color: blue; font-size: 30px;'>Skin Thickness Value Graph (Others vs Yours)</span>", unsafe_allow_html=True)

fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
#st.header(':blue[Insulin Value Graph (Others vs Yours)]')
st.markdown("<span style='color: blue; font-size: 30px;'>Insulin Value Graph (Others vs Yours)</span>", unsafe_allow_html=True)

fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
#st.header(':blue[BMI Value Graph (Others vs Yours)]')
st.markdown("<span style='color: blue; font-size: 30px;'>BMI Value Graph (Others vs Yours)</span>", unsafe_allow_html=True)

fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header(':blue[DPF Value Graph (Others vs Yours)]')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT
#st.subheader(':blue[Your Report: ]')
#output=''
#if user_result[0]==0:
#  output = ' :green[You are not Diabetic]'
#else:
#  output = ':red[You are Diabetic]'
#st.title(output)
#st.subheader(':orange[Accuracy: ]')
#st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')


st.markdown("<h2 style='text-align: center; color: blue;'>Your Report:</h2>", unsafe_allow_html=True)
if user_result[0]==0:
  st.markdown("<h2 style='text-align: center; color: green;'>You are not Diabetic</h2>", unsafe_allow_html=True)
else:
  st.markdown("<h2 style='text-align: center; color: red;'>You are Diabetic</h2>", unsafe_allow_html=True)
res = str(accuracy_score(y_test, rf.predict(x_test))*100)+'%'
st.markdown("<h5 style='text-align: center; color: purple;'>Accuracy : "+res+"</h5>", unsafe_allow_html=True)
