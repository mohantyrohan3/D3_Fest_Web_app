import streamlit as st
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
import pickle
# """Styling Fonts"""

# streamlit_style = """
# 			<style>
# 			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

# 			html, body, [class*="css"]  {
# 			font-family: 'Roboto', sans-serif;
# 			}
# 			</style>
# 			"""
# st.markdown(streamlit_style, unsafe_allow_html=True)

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

def predict(year,month,day,hour,temp,pressure,rain,wind_speed,wind_dir):
    model = pickle.load(open('model'+year, 'rb'))

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
        ans=predict(year,month,day,hour,temp,pressure,rain,wind_speed,wind_dir)
        ans=int(ans[0])
        st.success('The PM2.5 is '+str(ans))
        #st.write("The PM2.5 is ",ans[0])
    else:
        pass

elif(option=="Deep Learning"):
    year=st.selectbox("Select a year",('2013','2014','2015','2016','2017'))
    month=st.slider("month",min_value=1,max_value=12,step=1)
    day=st.slider("day",min_value=1,max_value=31,step=1)
    hour=st.slider("hours",min_value=0,max_value=23,step=1)
    temp=st.slider("temperature",min_value=-18,max_value=42,step=1)
    pressure=st.slider("pressure",min_value=0,max_value=1000,step=1)
    rain=st.slider("rain",min_value=0,max_value=72,step=1)
    wind_speed=st.slider("wind-speed",min_value=0,max_value=10,step=1)
    wind_dir=st.selectbox('Select the direction of wind',('E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'))
    pr=st.button("Predict")
    # if():
    #     st.sucess()
    # else:
    #     st.warning()
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











    