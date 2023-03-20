# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

#Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)



@st.cache
def predict_model(model,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
    glass_type=model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
    glass_type=glass_type[0]
    if glass_type==1:
        return ("Class 1: used for making building windows (float processed)")
    elif glass_type==2:
        return("Class 2: used for making building windows (non-float processed)")
    elif glass_type==3:
        return("Class 3: used for making vehicle windows (float processed)")
    elif glass_type==4:
        return("Class 4: used for making vehicle windows (non-float processed)")
    elif glass_type==5:
        return("Class 5: used for making containers")
    elif glass_type==6:
        return("Class 6: used for making tableware")
    elif glass_type==7:  
        return("Class 7: used for making headlamps")
st.title("GlassType Data analysis")
st.text('''This web app allows a user to predict the type of Glass 
         based on the parameters of Na(Sodium),Mg(Magnesium),Al(Aluminium),Si(Silicon),K(Potassium),Ca(Calcium),Ba(Barium),Fe(Iron).
         This Model also helps in visualization using different plots.''')

st.sidebar.subheader("glass_type detection app")
if st.sidebar.checkbox("show raw data"):
    st.subheader("full dataset")
    st.dataframe(glass_df)
st.sidebar.subheader("feature select")
feature_list=st.sidebar.multiselect( "SELECt X AXIS VALUE",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
viauliaser=st.sidebar.subheader("Visualisation Selector")
plot_list = st.sidebar.multiselect("Select the Charts/Plots:",
                                   ('histogram','scatter_plot','boxplot' , 'Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart'))

st.set_option('deprecation.showPyplotGlobalUse', False)

#display
if 'scatter_plot' in plot_list:
    for i in feature_list:
      st.subheader("scatter_plot between features and glass_type")
      plt.figure(figsize=(13,5))
      plt.scatter(glass_df[i],glass_df['GlassType'])
      st.pyplot()
if 'boxplot' in plot_list:
    for i in feature_list:
      st.subheader("boxplot")
      plt.figure(figsize=(13,5))
      sns.boxplot(glass_df[i])
      st.pyplot()
if 'histogram' in plot_list:
    for i in feature_list:
      st.subheader("scatter_plot between features and glass_type")
      plt.figure(figsize=(13,5))
      plt.hist(glass_df[i],bins='sturges')
      st.pyplot()

if 'Correlation Heatmap' in plot_list:
      st.subheader("Heatmap ")
      plt.figure(figsize=(13,5))
      sns.heatmap(glass_df.corr(),annot=True,cmap='jet')
      st.pyplot()
  
if 'Line Chart' in plot_list:
      st.subheader("Lineplot")
      st.line_chart(glass_df)
if 'Area Chart' in plot_list:
      st.subheader("AreaChart")
      st.area_chart(glass_df)
      
if 'Count Plot' in plot_list:
      st.subheader("Countplot")
      plt.figure(figsize=(13,5))
      sns.countplot(glass_df['GlassType'])
      st.pyplot()
if 'Pie Chart' in plot_list:
      st.subheader("scatter_plot between features and glass_type")
      plt.figure(figsize=(13,5))
      plt.pie(glass_df['GlassType'].value_counts(),labels=glass_df['GlassType'].value_counts().index,autopct='%1.1f')
      st.pyplot()


#model

st.sidebar.title("Glass type prediction model")
RI=st.sidebar.slider("RI",float(glass_df['RI'].min()),float(glass_df['RI'].max()))
Na=st.sidebar.slider("Na",float(glass_df['Na'].min()),float(glass_df['Na'].max()))
Mg=st.sidebar.slider("Mg",float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
Al=st.sidebar.slider("Al;",float(glass_df['Al'].min()),float(glass_df['Al'].max()))
Si=st.sidebar.slider("Si",float(glass_df['Si'].min()),float(glass_df['Si'].max()))
K=st.sidebar.slider("K",float(glass_df['K'].min()),float(glass_df['K'].max()))
Ca=st.sidebar.slider("Ca",float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
Ba=st.sidebar.slider("Ba",float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
Fe=st.sidebar.slider("Fe",float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier', 'Random Forest Classifier1','Support Vector Machine1','LogisticRegression1'))
if st.sidebar.button("Predict"):
    if classifier =='Suport Vector Machine':
       species_type=predict_model(svc_model,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
       score=svc_model.score(X_train,y_train)
       st.write("Species_predicted=",species_type)
       st.write("Accuracy of model is:",score)

if classifier=='Random Forest Classifier':
        species_type=predict_model(rf_clf,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        score=log_reg.score(X_train,y_train)
        st.write("Species_predicted=",species_type)
        st.write("Accuracy of model is:",score)
        
else:
        species_type=predict_model(log_reg,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        score=rf_clf.score(X_train,y_train)
        st.write("Species_predicted=",species_type)
        st.write("Accuracy of model is:",score)
    
from sklearn.metrics import plot_confusion_matrix

# if classifier =='Support Vector Machine', ask user to input the values of 'C','kernel' and 'gamma'.
if classifier == 'Support Vector Machine1':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = predict_model(svc_model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()
if classifier =='Random Forest Classifier1':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement. 
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf= RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = predict_model(rf_clf,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe )
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rf_clf, X_test, y_test)
        st.pyplot()
if classifier=='LogisticRegression1':
    st.sidebar.subheader("Model hyperparameters")
    c_value=st.sidebar.number_input("C",1,100,step=1)
    max_iter_input=st.sidebar.slider('Max_iter',10,1000,step=1)
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        log_reg=LogisticRegression(C = c_value, max_iter = max_iter_input)
        log_reg.fit(X_train,y_train)
        accuracy = log_reg.score(X_test, y_test)
        glass_type = predict_model(log_reg,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe )
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(log_reg, X_test, y_test)
        st.pyplot()


