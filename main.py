import streamlit as st
import tensorflow as tf
import numpy as np
import random
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("🌱 Plant Disease Detection System for Sustainable Agriculture 🌱")
st.sidebar.markdown(
    "A smart solution for sustainable agriculture using AI-powered disease detection."
)
app_mode = st.sidebar.selectbox("Navigate",["HOME","DISEASE RECOGNITION","ABOUT"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown(
        "<p style='text-align: center; font-size: 22px;'>"
        " 🌿 Welcome to the Plant Disease Detection System! Upload an image of a plant leaf, and let our AI detect potential diseases. 🌿"
        "</p>",
        unsafe_allow_html=True,
    )
    #st.title("ℹ️ Introduction :")
    st.markdown(
        """
        Our innovative AI-powered system is designed to revolutionize agriculture by providing early detection of plant diseases.
        By simply uploading an image of a plant leaf, farmers, researchers, and agricultural enthusiasts can identify potential diseases and take proactive measures to protect crops.
        
        ### Why Choose Our System?
        - Harnesses advanced AI technology for accurate predictions.
        - Supports a variety of plant species and diseases.
        - Promotes sustainable agriculture by minimizing crop loss and reducing manual labor.

        Empowering farmers and ensuring healthy crops is our mission. Together, let’s cultivate a greener future! 🌱.
        """
    )
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                    'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy', 
                    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))


    
    
    
    
    
    
    #feedback = st.text_area("Provide Feedback:")
    #if st.button("Submit Feedback"):
        #if feedback:
           # st.success("Thank you for your feedback!")
        #else:
            #st.warning("Please enter feedback before submitting.")


    # Example IoT data
    if st.button("Fetch IoT Data"):
        temperature = random.uniform(20, 35)
        humidity = random.uniform(40, 80)
        st.metric("Temperature (°C)", f"{temperature:.2f}")
        st.metric("Humidity (%)", f"{humidity:.2f}")



# About Page
elif (app_mode == "ABOUT"):
    st.title("ℹ️ About")
    st.markdown(
        """
        This system uses a Convolutional Neural Network (CNN) model to identify plant diseases based on leaf images. 
        By detecting diseases early, farmers can take preventive measures to ensure healthy crop growth and sustainable agriculture.
        
        ### Features:
        - Accurate plant disease prediction.
        - Support for a wide range of plant species and diseases.
        - IoT integration for real-time environment data.

        Developed with ❤️ by Yash Jondhale.
        """
    )

# Sidebar Extras
st.sidebar.markdown("---")
feedback = st.sidebar.text_area("💬 Feedback")
if st.sidebar.button("Submit Feedback"):
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.warning("Please provide feedback before submitting.")














