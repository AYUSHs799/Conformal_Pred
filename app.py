import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pickle

load_reg_data = False
load_class_data = False
california_img=plt.imread("./california.png")


# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.left'] = False
# plt.rcParams['axes.spines.bottom'] = False



def conformal_Predict(cal_err,alpha = 0.8):

    assert alpha != None, " Provide a value of alpha "
    # assert cal_err!= [] or None, "Provide the caliberation errors"
    idx = int(alpha*len(cal_err))
    return cal_err[idx]
    


if __name__ == '__main__':

    st.set_page_config(layout="wide")

    if not(load_reg_data):
        x_test = np.load("./Reg_Test_X.npy")
        y_test = np.load("./Reg_Test_y.npy")
        err_calib = np.load("./Reg_calib_err.npy")
        y_pred = np.load("./Reg_y_pred.npy")
        load_reg_data = True


    if not(load_class_data):
        pass



    st.title("Conformal Prediction")
    intro_tab , reg_tab , class_tab = st.tabs(["Introduction","Regression", "Classification"])
    css = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:2rem;
            }
        </style>
        '''

    st.markdown(css, unsafe_allow_html=True)

    with intro_tab:

        f = open("Introduction.md",'r')
        st.markdown(f.read())


    with reg_tab:
        with st.container():

            left,right = st.columns([3,2])

            with left:

                st.write(" ")

                st.markdown("For Regression, we are using California Housing Dataset. It serves as an excellent introduction to implementing machine learning algorithms because it has an easily understandable list of variables and sits at an optimal size between being too toyish and too cumbersome. The dataset pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data.")

                st.write("---")

                st.markdown("Lets assume you are a buyer in California who is intrested in buying a house. You will most likely have a budget in mind. We have trained a Random forest regressor on the california dataset that predicts the price of a property. This model will help you determine where you will be able to buy a house in california given your budget estimates")

                budget = st.slider('Your Budget (in Millions)',min_value=0.3,max_value=5.0,value=2.0,step=0.05)

                st.markdown("Now, Please select how certain you want the model to be. More the value of alpha, more certain the model will be and hence more accurate will the reading be for the price estimate")

                alpha = st.slider(' Select a value of alpha',min_value=0.1,max_value=.99,value=0.5,step=0.05)

                st.markdown("The green points indicate that your budget it greater than the upper bound of model's prediction and hence these properties could be bought. Red points however, are the are the areas where you wont be able to buy a house")

            with right:

                sigma = conformal_Predict(err_calib,1-alpha)
                in_range = (y_pred+sigma)<budget

                fig1, ax1 = plt.figure(figsize=(10,7),dpi=150), plt.gca()
                ax1.imshow(california_img, alpha=0.3,cmap=plt.get_cmap("jet"),extent=[-124.55, -113.80, 32.45, 42.05],zorder=1)
                ax1.scatter(x_test[in_range,7],x_test[in_range,6],s=10,alpha=0.5,label='Can be Bought',c='C2',zorder=3)
                ax1.scatter(x_test[~in_range,7],x_test[~in_range,6],s=10,alpha=0.5,label='Cannot Buy',c='r',zorder=3)

                
                
                ax1.set_title("California Housing Locations (Test-set)")
                ax1.set_xlabel("Latitude")
                ax1.set_ylabel("Longitude")
                ax1.spines['top'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_visible(False)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.legend()
                ax1.patch.set_alpha(0.0)
                st.pyplot(fig1)



    with class_tab:
        st.subheader("Classification")
        st.write("This is the classification tab")




