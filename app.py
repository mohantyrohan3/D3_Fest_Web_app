import streamlit as st
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -- Reading Dataset --
df=pd.read_csv('train.csv')
final_df=pd.read_csv("final_train.csv")



# Side Bar
ans=st.sidebar.selectbox("Select Graph",("Correlation Map","Dist Plot"))
if ans== 'Correlation Map':
    cor=st.sidebar.selectbox("Select Parameters for Correlation",("Rain","Wind Speed","Pressure"))
    if cor=="Rain":
        y = final_df['PM2.5']
        x = final_df['rain']
        correlation = y.corr(x)
        plt.title('Correlation between rain and PM 2.5')
        plt.xlim(-1, 50)
        plt.ylim(-10, 1000)
        f=plt.scatter(x, y)
        plt.plot()
        f=f.figure
        st.sidebar.pyplot(f)
    if cor=="Wind Speed":
        y = final_df['PM2.5']
        x = final_df['wind_speed']
        correlation = y.corr(x)
        plt.title('Correlation between Wind Speed and PM 2.5')
        plt.xlim(0,10)
        plt.ylim(0, 1000)
        f=plt.scatter(x, y)
        plt.plot()
        f=f.figure
        st.sidebar.pyplot(f)
    if cor=="Pressure":
        y = final_df['PM2.5']
        x = final_df['pressure']
        correlation = y.corr(x)
        plt.title('Correlation between Pressure and PM 2.5')
        plt.xlim(975,1050)
        plt.ylim(-1, 1000)
        f=plt.scatter(x, y)
        plt.plot()
        f=f.figure
        st.sidebar.pyplot(f)



elif ans=="Dist Plot":
    dis=st.sidebar.selectbox("Select Parameters for Dist Plot",("Wind Speed","Rain","Temperature"))
    if dis=="Wind Speed":
        f=sns.distplot(final_df['wind_speed'])
        f=f.figure
        st.sidebar.pyplot(f)
    if dis=="Rain":
        f=sns.distplot(final_df['rain'])
        f=f.figure
        st.sidebar.pyplot(f)
    if dis=="Temperature":
        f=sns.distplot(final_df['temperature'])
        f=f.figure
        st.sidebar.pyplot(f)








# """URL for Animation"""
def load_lottieurl(url):
     r = requests.get(url)
     if r.status_code != 200:
         return None
     return r.json()

lottie_coding=load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_EyJRUV.json")





# """Introduction Section"""
with st.container():
    components.html(""" 
    <div >
        <h1 style="text-align:center;color:white;font-family:monospace;font-size:50px;">
        PM 2.5
        </h1>
    </div
    """)

    left_column,right_column=st.columns([3,2])
    with left_column:
        
        st.subheader("What is PM ?")
        st.write(""" 
            PM stands for particulate matter (also called particle pollution): the term for a mixture of solid particles and liquid droplets found in the air. Some particles, such as dust, dirt, soot, or smoke, are large or dark enough to be seen with the naked eye. Others are so small they can only be detected using an electron microscope.
        """)
        st.subheader("What is PM 2.5?")
        st.write(""" 
            PM 2.5 refers to fine inhalable particles,with diameters that are generally 2.5 micrometers and smaller.
            How small is 2.5 micrometers? Think about a single hair from your head. The average human hair is about 70 micrometers in diameter – making it 30 times larger than the largest fine particle.
        """)

    with right_column:
        st_lottie(lottie_coding,height=500,key="air pollution")



st.write("---")




# """Prediction Part"""
wind={'E':7,'ENE':8,'ESE':9,'N':10,'NE':11,'NNE':12,'NNW':13,'NW':14,'S':15,'SE':16,'SSE':17,'SSW':18,'SW':19,'W':20,'WNW':21,'WSW':22}

def predict_old(year,month,day,hour,temp,pressure,rain,wind_speed,wind_dir):
    model = pickle.load(open('model'+year, 'rb'))

    inp=[float(month),float(day),float(hour),float(temp),float(pressure),float(rain),float(wind_speed)]
    for i in range(0,16):
        inp.append(0)
    
    inp[wind[wind_dir]]=1

    ans=model.predict([inp])
    return ans


def predict_new(year,month,day,hour,temp,pressure,rain,wind_speed,wind_dir):
    model = pickle.load(open('tuned_model'+year, 'rb'))

    inp=[float(month),float(day),float(hour),float(temp),float(pressure),float(rain),float(wind_speed)]
    for i in range(0,16):
        inp.append(0)
    
    inp[wind[wind_dir]]=1

    ans=model.predict([inp])
    return ans



st.title("PM  2.5 Prediction")
option = st.selectbox(
    'How would you like the model to be represented ?',
    ( 'Pick One ','Machine Learning', 'Deep Learning'))

if(option=="Machine Learning"):
    d=st.selectbox("Select Before or After Tuning ",("Not Tuned Model","Tuned Model",))
    year=st.selectbox("Select a year",('2013','2014','2015','2016','2017'))
    month=st.slider("month",min_value=1,max_value=12,step=1)
    day=st.slider("day",min_value=1,max_value=31,step=1)
    hour=st.slider("hours",min_value=0,max_value=23,step=1)
    temp=st.slider("temperature",min_value=-13,max_value=41,step=1)
    pressure=st.slider("pressure",min_value=900,max_value=1040,step=1)
    rain=st.slider("rain",min_value=0,max_value=40,step=1)
    wind_speed=st.slider("wind-speed",min_value=0,max_value=10,step=1)
    wind_dir=st.selectbox('Select the direction of wind',('E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'))
    
    
    pr=st.button("Predict")
    if(pr):
        if d=='Not Tuned Model':
            ans=predict_old(year,month,day,hour,temp,pressure,rain,wind_speed,wind_dir)
            ans=int(ans[0])
            st.success('The PM2.5 is '+str(ans))
            st.write("Details about the Model")
            st.write(" Model Used :  Random Forest Regressor")
        
        else:
            ans=predict_new(year,month,day,hour,temp,pressure,rain,wind_speed,wind_dir)
            ans=int(ans[0])
            st.success('The PM2.5 is '+str(ans))
            st.write("Details about the Model")
            st.write(" Model Used :  Random Forest Regressor")
        #st.write("The PM2.5 is ",ans[0])
    else:
        pass

elif(option=="Deep Learning"):
    
    lottie_coding_dl=load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_vw2szd2m.json')
    st_lottie(lottie_coding_dl,height=300)
else:
    st.write("You Haven't Selected Any Option")



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.write("---")



# """Graph Part"""
st.subheader("PM2.5 Dataset")
st.write(df)


if st.button("Intercorrelation Heatmap"):
    st.subheader("Intercorrelation Heatmap")
    corrmat = df.corr()
    top_corr_feature = corrmat.index
    plt.figure(figsize = (5,5))
    g = sns.heatmap(df[top_corr_feature].corr(), annot=True,cmap='viridis')
    g=g.figure
    st.pyplot(g)





# Footer
st.write("---")

thnk=load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_sqpjokxl.json')
st_lottie(thnk,height=300)