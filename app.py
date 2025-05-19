
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import cv2
import tempfile
import time  # Import the time module

import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv

from googletrans import Translator  # Import the Translator class

warnings.filterwarnings('ignore')

# --- 0. Disease Class Names ---
disease_class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 0.1. Treatment Rules Dictionary (Now with environmental context) ---
treatment_rules = {
    'Apple___Apple_scab': {
        'en': {'default': 'Chemical: Apply a fungicide containing captan or mancozeb. Organic: Improve air circulation, remove fallen leaves.',
               'high_humidity': 'Consider a systemic fungicide if humidity remains high.'},
        'hi': {'default': 'रासायनिक: कैप्टन या मैनकोजेब युक्त कवकनाशी लगाएं। जैविक: हवा का संचार बेहतर करें, गिरे हुए पत्तों को हटा दें।',
               'high_humidity': 'यदि नमी अधिक बनी रहती है तो एक प्रणालीगत कवकनाशी पर विचार करें।'}},
    'Apple___Black_rot': {
        'en': {'default': 'Chemical: Prune infected branches. Use a fungicide containing copper or myclobutanil. Organic: Remove mummified fruit, improve drainage.',
               'wet_conditions': 'Ensure good drainage to prevent further spread in wet conditions.'},
        'hi': {'default': 'रासायनिक: संक्रमित शाखाओं को छाँटें। कॉपर या मायक्लोबुटानिल युक्त कवकनाशी का प्रयोग करें। जैविक: ममीकृत फलों को हटा दें, जल निकासी में सुधार करें।',
               'wet_conditions': 'गीली परिस्थितियों में आगे प्रसार को रोकने के लिए अच्छी जल निकासी सुनिश्चित करें।'}},
    # Add more rules with environmental context for other diseases
    'Apple___Cedar_apple_rust': {
        'en': {'default': 'Chemical: Remove cedar trees within a certain radius. Apply a fungicide like myclobutanil or propiconazole. Organic: Remove galls from cedar trees.',
               'rainy_spring': 'Apply preventative fungicides during rainy spring periods.'},
        'hi': {'default': 'रासायनिक: एक निश्चित दायरे के भीतर देवदार के पेड़ों को हटा दें। मायक्लोबुटानिल या प्रोपिकोनाजोल जैसे कवकनाशी का प्रयोग करें। जैविक: देवदार के पेड़ों से पित्त को हटा दें।',
               'rainy_spring': 'बरसात के वसंत के दौरान निवारक कवकनाशी लगाएं।'}},
    'Apple___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Blueberry___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Cherry_(including_sour)___Powdery_mildew': {
        'en': {'default': 'Chemical: Apply a fungicide containing myclobutanil or sulfur. Organic: Improve air circulation, use neem oil.',
               'dry_weather': 'Powdery mildew thrives in dry conditions; ensure consistent watering.'},
        'hi': {'default': 'रासायनिक: मायक्लोबुटानिल या सल्फर युक्त कवकनाशी लगाएं। जैविक: हवा का संचार बेहतर करें, नीम के तेल का प्रयोग करें।',
               'dry_weather': 'पाउडरी मिल्ड्यू शुष्क परिस्थितियों में पनपता है; लगातार पानी देना सुनिश्चित करें।'}},
    'Cherry_(including_sour)___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'en': {'default': 'Chemical: Apply a fungicide containing azoxystrobin or propiconazole. Organic: Crop rotation, improve drainage.',
               'warm_humid': 'These leaf spots are favored by warm and humid conditions.'},
        'hi': {'default': 'रासायनिक: एज़ोक्सीस्ट्रोबिन या प्रोपिकोनाजोल युक्त कवकनाशी लगाएं। जैविक: फसल चक्रण, जल निकासी में सुधार करें।',
               'warm_humid': 'ये पत्ती धब्बे गर्म और आर्द्र परिस्थितियों में पनपते हैं।'}},
    'Corn_(maize)___Common_rust_': {
        'en': {'default': 'Chemical: Apply a fungicide containing azoxystrobin or propiconazole. Organic: Plant resistant varieties.',
               'moderate_temp_high_humidity': 'Rust development is favored by moderate temperatures and high humidity.'},
        'hi': {'default': 'रासायनिक: एज़ोक्सीस्ट्रोबिन या प्रोपिकोनाजोल युक्त कवकनाशी लगाएं। जैविक: प्रतिरोधी किस्मों को लगाएं।',
               'moderate_temp_high_humidity': 'रस्ट का विकास मध्यम तापमान और उच्च आर्द्रता सेfavor होता है।'}},
    'Corn_(maize)___Northern_Leaf_Blight': {
        'en': {'default': 'Chemical: Apply a fungicide containing azoxystrobin or propiconazole. Organic: Crop rotation, remove infected debris.',
               'wet_weather': 'Blight can spread quickly in wet weather.'},
        'hi': {'default': 'रासायनिक: एज़ोक्सीस्ट्रोबिन या प्रोपिकोनाजोल युक्त कवकनाशी लगाएं। जैविक: फसल चक्रण, संक्रमित मलबे को हटा दें।',
               'wet_weather': 'गीले मौसम में ब्लाइट तेजी से फैल सकता है।'}},
    'Corn_(maize)___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Grape___Black_rot': {
        'en': {'default': 'Chemical: Apply a fungicide containing captan or myclobutanil. Organic: Remove infected berries, improve air circulation.',
               'frequent_rain': 'Black rot is more severe with frequent rainfall.'},
        'hi': {'default': 'रासायनिक: कैप्टन या मायक्लोबुटानिल युक्त कवकनाशी लगाएं। जैविक: संक्रमित जामुन को हटा दें, हवा का संचार बेहतर करें।',
               'frequent_rain': 'बार-बार बारिश होने पर ब्लैक रॉट अधिक गंभीर होता है।'}},
    'Grape___Esca_(Black_Measles)': {
        'en': {'default': 'Chemical: There is no effective chemical control. Organic: Prune infected wood, sanitize tools.',
               'wet_spring': 'Consider preventative pruning, especially after wet springs.'},
        'hi': {'default': 'रासायनिक: कोई प्रभावी रासायनिक नियंत्रण नहीं है। जैविक: संक्रमित लकड़ी को छाँटें, उपकरणों को साफ करें।',
               'wet_spring': 'विशेष रूप से गीली स्प्रिंग्स के बाद निवारक छंटाई पर विचार करें।'}},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'en': {'default': 'Chemical: Apply a copper-based fungicide. Organic: Remove infected leaves, improve air circulation.',
               'cool_wet': 'Leaf blight favors cool and wet conditions.'},
        'hi': {'default': 'रासायनिक: कॉपर आधारित कवकनाशी लगाएं। जैविक: संक्रमित पत्तों को हटा दें, हवा का संचार बेहतर करें।',
               'cool_wet': 'पत्ती झुलसा ठंडी और गीली परिस्थितियों को पसंद करता है।'}},
    'Grape___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Orange___Haunglongbing_(Citrus_greening)': {
        'en': {'default': 'Chemical: No effective chemical control.  Organic: Remove infected trees, control psyllid vectors.',
               'warm_climate': 'The psyllid vector thrives in warm climates.'},
        'hi': {'default': 'रासायनिक: कोई प्रभावी रासायनिक नियंत्रण नहीं है। जैविक: संक्रमित पेड़ों को हटा दें, साइलिड वैक्टर को नियंत्रित करें।',
               'warm_climate': 'साइलिड वेक्टर गर्म जलवायु में पनपता है।'}},
    'Peach___Bacterial_spot': {
        'en': {'default': 'Chemical: Apply copper-based bactericides. Organic: Use disease-free nursery stock, improve air circulation.',
               'high_humidity_warm_temp': 'Bacterial spot is favored by high humidity and warm temperatures.'},
        'hi': {'default': 'रासायनिक: कॉपर आधारित जीवाणुनाशक लगाएं। जैविक: रोग-मुक्त नर्सरी स्टॉक का उपयोग करें, हवा का संचार बेहतर करें।',
               'high_humidity_warm_temp': 'उच्च आर्द्रता और गर्म तापमान से बैक्टीरियल स्पॉट favor होता है।'}},
    'Peach___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Pepper,_bell___Bacterial_spot': {
        'en': {'default': 'Chemical: Use copper-based fungicides. Avoid overhead watering. Organic: Use disease-free seeds, crop rotation.',
               'warm_wet': 'Warm and wet conditions favor bacterial spot.'},
        'hi': {'default': 'रासायनिक: कॉपर आधारित कवकनाशी का प्रयोग करें। ऊपर से पानी देना ഒഴിവാ করুন। जैविक: रोग-मुक्त बीजों का प्रयोग करें, फसल चक्रण करें।',
               'warm_wet': 'गर्म और गीली परिस्थितियाँ बैक्टीरियल स्पॉट को favor करती हैं।'}},
    'Pepper,_bell___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Potato___Early_blight': {
        'en': {'default': 'Chemical: Apply a fungicide containing chlorothalonil or mancozeb. Organic: Improve soil health, remove infected leaves.',
               'warm_dry_periods_with_dew': 'Early blight often occurs during warm, dry periods with heavy dews.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या मैनकोजेब युक्त कवकनाशी लगाएं। जैविक: मिट्टी के स्वास्थ्य में सुधार करें, संक्रमित पत्तों को हटा दें।',
               'warm_dry_periods_with_dew': 'अगेती झुलसा अक्सर भारी ओस के साथ गर्म, शुष्क अवधि के दौरान होता है।'}},
    'Potato___Late_blight': {
        'en': {'default': 'Chemical: Use a fungicide containing chlorothalonil or protectant fungicides like mancozeb. Organic: Improve air circulation, use resistant varieties.',
               'cool_wet_weather': 'Late blight thrives in cool, wet weather.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या मैनकोजेब जैसे सुरक्षात्मक कवकनाशी का प्रयोग करें। जैविक: हवा का संचार बेहतर करें, प्रतिरोधी किस्मों का प्रयोग करें।',
               'cool_wet_weather': 'पिछेती झुलसा ठंडे, गीले मौसम में पनपता है।'}},
    'Potato___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Raspberry___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Soybean___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Squash___Powdery_mildew': {
        'en': {'default': 'Chemical: Apply a fungicide containing myclobutanil or sulfur. Organic: Improve air circulation, use neem oil or baking soda spray.',
               'dry_shade': 'Powdery mildew can be worse in dry, shady areas.'},
        'hi': {'default': 'रासायनिक: मायक्लोबुटानिल या सल्फर युक्त कवकनाशी लगाएं। जैविक: हवा का संचार बेहतर करें, नीम के तेल या बेकिंग सोडा स्प्रे का प्रयोग करें।',
               'dry_shade': 'पाउडरी मिल्ड्यू शुष्क, छायादार क्षेत्रों में खराब हो सकता है।'}},
    'Strawberry___Leaf_scorch': {
        'en': {'default': 'Chemical: Apply a fungicide containing captan or myclobutanil. Organic: Remove infected leaves, improve air circulation.',
               'warm_humid_weather': 'Leaf scorch is favored by warm, humid weather.'},
        'hi': {'default': 'रासायनिक: कैप्टन या मायक्लोबुटानिल युक्त कवकनाशी लगाएं। जैविक: संक्रमित पत्तों को हटा दें, हवा का संचार बेहतर करें।',
               'warm_humid_weather': 'गर्म, आर्द्र मौसम से पत्ती झुलसा favor होता है।'}},
    'Strawberry___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}},
    'Tomato___Bacterial_spot': {
        'en': {'default': 'Chemical: Use copper-based fungicides. Avoid overhead watering. Organic: Use disease-free seeds, crop rotation.',
               'warm_humid_rainy': 'Bacterial spot spreads easily in warm, humid, and rainy conditions.'},
        'hi': {'default': 'रासायनिक: कॉपर आधारित कवकनाशी का प्रयोग करें। ऊपर से पानी देना ഒഴിവാ করুন। जैविक: रोग-मुक्त बीजों का प्रयोग करें, फसल चक्रण करें।',
               'warm_humid_rainy': 'गर्म, आर्द्र और बरसात की परिस्थितियों में बैक्टीरियल स्पॉट आसानी से फैलता है।'}},
    'Tomato___Early_blight': {
        'en': {'default': 'Chemical: Apply a fungicide containing chlorothalonil or mancozeb. Prune lower infected leaves. Organic: Improve soil health, remove infected leaves.',
               'alternating_wet_dry': 'Early blight can be more severe with alternating wet and dry periods.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या मैनकोजेब युक्त कवकनाशी लगाएं। निचले संक्रमित पत्तों को छाँटें। जैविक: मिट्टी के स्वास्थ्य में सुधार करें, संक्रमित पत्तों को हटा दें।',
               'alternating_wet_dry': 'अगेती झुलसा गीली और शुष्क अवधि केalternating होने पर अधिक गंभीर हो सकता है।'}},
    'Tomato___Late_blight': {
        'en': {'default': 'Chemical: Use a fungicide containing chlorothalonil or protectant fungicides like mancozeb. Improve air circulation. Organic: Use resistant varieties, improve air circulation.',
               'cool_wet_foggy': 'Late blight favors cool, wet, and foggy weather.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या मैनकोजेब जैसे सुरक्षात्मक कवकनाशी का प्रयोग करें। हवा का संचार बेहतर करें। जैविक: प्रतिरोधी किस्मों का प्रयोग करें, हवा का संचार बेहतर करें।',
               'cool_wet_foggy': 'पिछेती झुलसा ठंडे, गीले और कोहरे वाले मौसम को पसंद करता है।'}},
    'Tomato___Leaf_Mold': {
        'en': {'default': 'Chemical: Apply a fungicide containing chlorothalonil or mancozeb. Organic: Improve air circulation, reduce humidity.',
               'high_humidity_low_airflow': 'Leaf mold thrives in high humidity and low air flow environments.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या मैनकोजेब युक्त कवकनाशी लगाएं।जैविक: हवा का संचार बेहतर करें, नमी कम करें।',
               'high_humidity_low_airflow': 'पत्ती मोल्ड उच्च आर्द्रता और कम वायु प्रवाह वाले वातावरण में पनपता है।'}},
    'Tomato___Septoria_leaf_spot': {
        'en': {'default': 'Chemical: Apply a fungicide containing chlorothalonil or mancozeb. Organic: Remove infected leaves, improve air circulation.',
               'warm_humid_with_rain': 'Septoria leaf spot is favored by warm, humid conditions with frequent rain.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या मैनकोजेब युक्त कवकनाशी लगाएं। जैविक: संक्रमित पत्तों को हटा दें, हवा का संचार बेहतर करें।',
               'warm_humid_with_rain': 'सेप्टोरिया लीफ स्पॉट गर्म, आर्द्र परिस्थितियों में बार-बार बारिश होने पर favor होता है।'}},
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'en': {'default': 'Chemical: Use miticides like abamectin or spiromesifen. Organic: Use insecticidal soap, neem oil, or introduce predatory mites.',
               'hot_dry': 'Spider mites thrive in hot, dry conditions.'},
        'hi': {'default': 'रासायनिक: एबामेक्टिन या स्पाइरोमेसिफेन जैसे माइटिसाइड का प्रयोग करें। जैविक: कीटनाशक साबुन, नीम का तेल प्रयोग करें, या शिकारी माइट्स को छोड़ें।',
               'hot_dry': 'स्पाइडर माइट्स गर्म, शुष्क परिस्थितियों में पनपते हैं।'}},
    'Tomato___Target_Spot': {
        'en': {'default': 'Chemical: Apply a fungicide containing chlorothalonil or copper. Organic: Remove infected leaves, improve air circulation.',
               'warm_humid': 'Target spot is favored by warm and humid conditions.'},
        'hi': {'default': 'रासायनिक: क्लोरोथालोनिल या कॉपर युक्त कवकनाशी लगाएं। जैविक: संक्रमित पत्तों को हटा दें, हवा का संचार बेहतर करें।',
               'warm_humid': 'टारगेट स्पॉट गर्म और आर्द्र परिस्थितियों में favor होता है।'}},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'en': {'default': 'Chemical: No effective chemical control for the virus itself.  Control whitefly vectors with insecticides. Organic: Remove infected plants, control whitefly vectors with insecticidal soap.',
               'warm_climate_whiteflies': 'The whitefly vector is more active in warm climates.'},
        'hi': {'default': 'रासायनिक: वायरस के लिए कोई प्रभावी रासायनिक नियंत्रण नहीं है। कीटनाशकों से व्हाइटफ्लाई वैक्टर को नियंत्रित करें। जैविक: संक्रमित पौधों को हटा दें, कीटनाशक साबुन से व्हाइटफ्लाई वैक्टर को नियंत्रित करें।',
               'warm_climate_whiteflies': 'व्हाइटफ्लाई वेक्टर गर्म जलवायु में अधिक सक्रिय है।'}},
    'Tomato___Tomato_mosaic_virus': {
        'en': {'default': 'Chemical: No effective chemical control for the virus itself. Organic: Remove infected plants, sanitize tools.',
               'no_direct_environmental_link': 'Spread is mainly through contact and vectors.'},
        'hi': {'default': 'रासायनिक: वायरस के लिए कोई प्रभावी रासायनिक नियंत्रण नहीं है। जैविक: संक्रमित पौधों को हटा दें, उपकरणों को साफ करें।',
               'no_direct_environmental_link': 'प्रसार मुख्य रूप से संपर्क और वैक्टर के माध्यम से होता है।'}},
    'Tomato___healthy': {'en': {'default': 'No treatment needed.'}, 'hi': {'default': 'किसी उपचार की आवश्यकता नहीं।'}}
}

# --- 1. Plant Disease Recognition Functions ---
@st.cache_resource
def load_disease_model():
    """Loads the TensorFlow model for disease prediction."""
    try:
        model = tf.keras.models.load_model('trained_model.h5') # Use absolute path
        return model
    except Exception as e:
        st.error(f"Error loading disease model: {e}")
        return None

disease_model = load_disease_model()
from keras.utils import load_img, img_to_array

def model_prediction(image_file):
    image = load_img(image_file, target_size=(128, 128))
    image_arr = img_to_array(image)  # ✅ no division
    image_arr = image_arr.reshape(1, 128, 128, 3)

    prediction = disease_model.predict(image_arr)
    predicted_index = np.argmax(prediction[0])
    predicted_disease = disease_class_names[predicted_index]

    return predicted_disease, prediction[0], image_arr

# --- 1.1. Treatment Recommendation Function (Now considers environment) ---
def get_treatment(predicted_disease, language, temperature=None, humidity=None, rainfall=None):
    """
    Retrieves the treatment recommendation for a given disease in the specified language,
    considering environmental conditions.

    Args:
        predicted_disease (str): The name of the predicted plant disease.
        language (str): The desired language ('en' for English, 'hi' for Hindi).
        temperature (float, optional): Current temperature.
        humidity (float, optional): Current humidity.
        rainfall (float, optional): Recent rainfall.

    Returns:
        str or None: The treatment recommendation if found, otherwise None.
    """
    if predicted_disease in treatment_rules:
        lang_rules = treatment_rules[predicted_disease].get(language, treatment_rules[predicted_disease]['en']) # Default to English if language not found
        treatment = lang_rules.get('default')
        if temperature is not None and humidity is not None and rainfall is not None:
            if 'high_humidity' in lang_rules and humidity > 70:
                treatment += f"\n{lang_rules['high_humidity']}"
            if 'low_humidity' in lang_rules and humidity < 30:
                treatment += f"\n{lang_rules['low_humidity']}"
            if 'wet_conditions' in lang_rules and rainfall > 5:
                treatment += f"\n{lang_rules['wet_conditions']}"
            if 'dry_weather' in lang_rules and rainfall < 1:
                treatment += f"\n{lang_rules['dry_weather']}"
            if 'warm_weather' in lang_rules and temperature > 30:
                treatment += f"\n{lang_rules['warm_weather']}"
            if 'cool_weather' in lang_rules and temperature < 15:
                treatment += f"\n{lang_rules['cool_weather']}"
            if 'rainy_spring' in lang_rules and rainfall > 1 and 3 <= time.localtime().tm_mon <= 5: # Spring months
                treatment += f"\n{lang_rules['rainy_spring']}"
            if 'dry_shade' in lang_rules and rainfall < 1:
                treatment += f"\n{lang_rules['dry_shade']}"
            if 'warm_humid' in lang_rules and temperature > 25 and humidity > 60:
                treatment += f"\n{lang_rules['warm_humid']}"
            if 'moderate_temp_high_humidity' in lang_rules and 18 <= temperature <= 25 and humidity > 70:
                treatment += f"\n{lang_rules['moderate_temp_high_humidity']}"
            if 'wet_weather' in lang_rules and rainfall > 3:
                treatment += f"\n{lang_rules['wet_weather']}"
            if 'cool_wet' in lang_rules and temperature < 20 and rainfall > 3:
                treatment += f"\n{lang_rules['cool_wet']}"
            if 'warm_climate' in lang_rules and temperature > 28:
                treatment += f"\n{lang_rules['warm_climate']}"
            if 'high_humidity_warm_temp' in lang_rules and humidity > 75 and temperature > 28:
                treatment += f"\n{lang_rules['high_humidity_warm_temp']}"
            if 'warm_wet' in lang_rules and temperature > 25 and rainfall > 5:
                treatment += f"\n{lang_rules['warm_wet']}"
            if 'warm_dry_periods_with_dew' in lang_rules and temperature > 22 and rainfall < 1 and humidity > 80: # Assuming dew implies high humidity
                treatment += f"\n{lang_rules['warm_dry_periods_with_dew']}"
            if 'cool_wet_foggy' in lang_rules and temperature < 20 and rainfall > 2 and humidity > 90: # Assuming fog implies very high humidity
                treatment += f"\n{lang_rules['cool_wet_foggy']}"
            if 'alternating_wet_dry' in lang_rules and rainfall > 1: # Simple check for some rain
                treatment += f"\n{lang_rules['alternating_wet_dry']}"
            if 'high_humidity_low_airflow' in lang_rules and humidity > 70: # Simple check for high humidity
                treatment += f"\n{lang_rules['high_humidity_low_airflow']}"
            if 'warm_humid_with_rain' in lang_rules and temperature > 25 and humidity > 65 and rainfall > 1:
                treatment += f"\n{lang_rules['warm_humid_with_rain']}"
            if 'hot_dry' in lang_rules and temperature > 32 and humidity < 40:
                treatment += f"\n{lang_rules['hot_dry']}"
            if 'warm_climate_whiteflies' in lang_rules and temperature > 28:
                treatment += f"\n{lang_rules['warm_climate_whiteflies']}"
            if 'no_direct_environmental_link' in lang_rules:
                treatment += f"\n{lang_rules['no_direct_environmental_link']}"

        return treatment
    else:
        return None

# ----3. Fertilizer Model
fert_model = pickle.load(open('fert_model_simple.pkl', 'rb'))
crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))
fertilizer_encoder = pickle.load(open('fertilizer_encoder.pkl', 'rb'))
def predict_fertilizer_simple(n, p, k, crop_name):
    try:
        crop_code = crop_encoder.transform([crop_name])[0]
        input_data = pd.DataFrame([[crop_code, n, p, k]], columns=['Crop', 'Nitrogen', 'Phosphorous', 'Potassium'])
        fert_code = fert_model.predict(input_data)[0]
        fert_name = fertilizer_encoder.inverse_transform([fert_code])[0]
        return fert_name
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# --- 2. Crop Recommendation Functions ---
# Load the model
try:
    model = pickle.load(open('crop_recommendation_model3.pkl','rb'))
except Exception as e:
    st.error(f"Failed to load crop recommendation model or scalers: {e}")
    st.stop()

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Dictionary for crop names in Hindi
crop_dict_hi = {
    "Rice": "चावल",
    "Maize": "मक्का",
    "Jute": "जूट",
    "Cotton": "कपास",
    "Coconut": "नारियल",
    "Papaya": "पपीता",
    "Orange": "संतरा",
    "Apple": "सेब",
    "Muskmelon": "खरबूजा",
    "Watermelon": "तरबूज",
    "Grapes": "अंगूर",
    "Mango": "आम",
    "Banana": "केला",
    "Pomegranate": "अनार",
    "Lentil": "मसूर दाल",
    "Blackgram": "उड़द दाल",
    "Mungbean": "मूंग दाल",
    "Mothbeans": "मोठ दाल",
    "Pigeonpeas": "अरहर दाल",
    "Kidneybeans": "राजमा",
    "Chickpea": "चना",
    "Coffee": "कॉफी"
}

fert_crop_dict_hi = {
    "Cotton": "कपास",
    "Sugarcane": "गन्ना",
    "Paddy": "धान",
    "Maize": "मक्का",
    "Pulses": "दालें",
    "Barley": "जौ",
    "Millets": "बाजरा",
    "Wheat": "गेहूं",
    "Ground Nuts": "मूंगफली",
    "Tobacco": "तंबाकू",
    "Oil seeds": "तिलहन",
    "Banana": "केला",
    "Lentil": "मसूर दाल",
    "Coffee": "कॉफी",
    "Jute": "जूट"
}



# --3 Crop Names in Fertlizer recommendation in Hindi
def get_crop_display_and_mapping(crop_options, lang_code):
    fert_crop_dict_hi = {
        "Cotton": "कपास",
        "Sugarcane": "गन्ना",
        "Paddy": "धान",
        "Maize": "मक्का",
        "Pulses": "दालें",
        "Barley": "जौ",
        "Millets": "बाजरा",
        "Wheat": "गेहूं",
        "Ground Nuts": "मूंगफली",
        "Tobacco": "तंबाकू",
        "Oil seeds": "तिलहन",
        "Banana": "केला",
        "Lentil": "मसूर दाल",
        "Coffee": "कॉफी",
        "Jute": "जूट"
    }
    if lang_code == 'hi':
        crop_display = [fert_crop_dict_hi.get(crop, crop) for crop in crop_options]
        crop_mapping = dict(zip(crop_display, crop_options))
    else:
        crop_display = crop_options
        crop_mapping = dict(zip(crop_display, crop_display))
    return crop_display, crop_mapping



def predict_crop(N, P, K, temperature, humidity, ph, rainfall, language):
    try:
        feature_list = [N, P, K, temperature, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        prediction = model.predict(single_pred)
        predicted_crop = prediction[0]
        
        # Normalize the casing of the predicted crop
        predicted_crop_normalized = predicted_crop.strip().title()  # Convert to title case

        # Log the predicted crop and its type for debugging in Streamlit
        st.write(f"Predicted crop: {predicted_crop_normalized}")

        # Case where the model returns crop name (string)
        if isinstance(predicted_crop_normalized, str):
            if predicted_crop_normalized in [crop.title() for crop in crop_dict.values()]:
                crop_en = predicted_crop_normalized
                if language == 'hi':
                    crop_hi = crop_dict_hi.get(crop_en, crop_en)
                    result = f"{crop_hi} यहाँ उगाने के लिए सबसे अच्छी फसल है"
                else:
                    result = f"{crop_en} is the best crop to be cultivated right there"
            else:
                st.write(f"Prediction is not a valid crop name: {predicted_crop_normalized}")
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
                if language == 'hi':
                    result = "क्षमा करें, हम दिए गए डेटा के साथ उगाने के लिए सबसे अच्छी फसल निर्धारित नहीं कर सके।"

        # Case where the model returns an integer (class index)
        elif isinstance(predicted_crop_normalized, int) and 1 <= predicted_crop_normalized <= 22:
            crop_en = crop_dict[predicted_crop_normalized]
            if language == 'hi':
                crop_hi = crop_dict_hi.get(crop_en, crop_en)
                result = f"{crop_hi} यहाँ उगाने के लिए सबसे अच्छी फसल है"
            else:
                result = f"{crop_en} is the best crop to be cultivated right there"
        else:
            st.write(f"Invalid prediction: {predicted_crop}")
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            if language == 'hi':
                result = "क्षमा करें, हम दिए गए डेटा के साथ उगाने के लिए सबसे अच्छी फसल निर्धारित नहीं कर सके।"

        return result
    except Exception as e:
        error_message = f"An error occurred during crop prediction: {e}"
        if language == 'hi':
            error_message = f"फसल की भविष्यवाणी के दौरान एक त्रुटि हुई: {e}"
        st.error(error_message)
        return None


# --- 3. Main Streamlit App ---

def explain_prediction_with_lime(image_array, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_array[0].astype('double'),  # ✅ same image used for prediction
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False  # ✅ This keeps the original image visible
)
    return temp, mask
def main():
    st.sidebar.title("भाषा चयन")
    language = st.sidebar.selectbox("भाषा चुनें / Choose Language", ["English", "हिंदी"])
    if language == "हिंदी":
        lang_code = 'hi'
    else:
        lang_code = 'en'

    translator = Translator()

    def translate_text(text, dest='hi'):
        if lang_code == 'hi':
            return translator.translate(text, dest=dest).text
        return text

    st.sidebar.title(translate_text("Dashboard", dest=lang_code))
    app_mode = st.sidebar.selectbox(
        translate_text("Select Page", dest=lang_code),
        ["Home", "About", "Disease Recognition", "Crop Recommendation", "Fertilizer Recommendation"]
    )

    if app_mode == "Home":
        st.header(translate_text("ANN DOOT", dest=lang_code))
        image_path = "home_page.jpeg"
        st.image(image_path, use_container_width=True)
        st.markdown(
            translate_text(
                """
                Welcome to the Plant Disease Recognition, Crop Recommendation System and Fertilizer Recommendation System! 🌿🔍

                Our mission is to help farmers and gardeners in identifying plant diseases, recommending suitable crops,
                and suggesting the best fertilizers based on environmental and soil conditions.

                How It Works
                1.  Plant Disease Recognition: Upload an image of a plant, or capture one using your camera, and our system will analyze
                    it to detect any signs of diseases and suggest potential treatments, now considering environmental factors.  
                2.  Crop Recommendation: Enter the environmental factors, and our system will suggest the most suitable crop
                    to cultivate.  
                3.  Fertilizer Recommendation: Based on the soil's nutrient levels (N, P, K) and the selected crop, the system suggests the most appropriate fertilizer to use.

                ### Get Started
                Select a page from the sidebar to explore the features!
                """, dest=lang_code)
        )

    elif app_mode == "About":
        st.header(translate_text("About Us", dest=lang_code))
        st.markdown(
            translate_text(
                """
                We are a team dedicated to developing solutions for plant disease recognition and crop recommendation.
                Our goal is to assist farmers and gardeners in making informed decisions to improve crop health and productivity by considering various data sources.
                """, dest=lang_code)
        )

    elif app_mode == "Disease Recognition":
        st.header(translate_text("Plant Disease Recognition", dest=lang_code))

        # ✅ Initialize variables before usage
        predicted_disease = None
        image_arr = None

        input_option = st.radio(
            translate_text("Choose input method:", dest=lang_code),
            [translate_text("Upload Image", dest=lang_code), translate_text("Capture Image", dest=lang_code)]
        )

        temperature_input = st.number_input(translate_text("Temperature (°C, optional)", dest=lang_code), min_value=-20.0, max_value=50.0, value=None)
        humidity_input = st.number_input(translate_text("Humidity (%, optional)", dest=lang_code), min_value=0.0, max_value=100.0, value=None)
        rainfall_input = st.number_input(translate_text("Rainfall (mm, optional)", dest=lang_code), min_value=0.0, max_value=500.0, value=None)

        if input_option == translate_text("Upload Image", dest=lang_code):
            test_image = st.file_uploader(
                translate_text("Choose an Image:", dest=lang_code),
                type=["png", "jpg", "jpeg"]
            )
            if st.button(translate_text("Show Image", dest=lang_code)) and test_image is not None:
                st.image(test_image, use_container_width=True)

            if st.button(translate_text("Predict", dest=lang_code)) and test_image is not None:
                with st.spinner(translate_text("Please Wait..", dest=lang_code)):
                    st.write(translate_text("Our Prediction", dest=lang_code))
                    predicted_disease, prediction_probabilities, image_arr = model_prediction(test_image)

        # ✅ Display results after prediction
        if predicted_disease:
            translated_prediction = translate_text(f"Model is Predicting it's a {predicted_disease}", dest=lang_code)
            st.success(translated_prediction)

            from skimage.segmentation import mark_boundaries

            # LIME Explanation
            st.subheader(translate_text("LIME Explanation", dest=lang_code))

            # Generate explanation
            temp, mask = explain_prediction_with_lime(image_arr, disease_model)

            # Use original image (not temp) for visualization
            original_image = image_arr[0].astype('uint8')  # recover original input

            fig, ax = plt.subplots()
            ax.imshow(mark_boundaries(original_image, mask))  # ✅ Use original input image here
            ax.axis('off')
            st.pyplot(fig)


            # Treatment
            treatment = get_treatment(predicted_disease, lang_code, temperature_input, humidity_input, rainfall_input)
            if treatment:
                st.info(treatment)
            else:
                st.warning(translate_text("No specific treatment recommendation available for this disease yet.", dest=lang_code))
                
        elif input_option == translate_text("Capture Image", dest=lang_code):
            captured_image = st.camera_input(translate_text("Capture an image of the plant:", dest=lang_code))
            if captured_image:
                st.image(captured_image, caption=translate_text("Captured Image", dest=lang_code), use_container_width=True)
                if st.button(translate_text("Predict", dest=lang_code)):
                    with st.spinner(translate_text("Please Wait..", dest=lang_code)):
                        st.write(translate_text("Our Prediction", dest=lang_code))
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                            temp_file.write(captured_image.read())
                            temp_file_path = temp_file.name

                        predicted_disease, prediction_probabilities, image_arr = model_prediction(temp_file_path)

                        if predicted_disease:
                            translated_prediction = translate_text(f"Model is Predicting it's a {predicted_disease}", dest=lang_code)
                            st.success(translated_prediction)

                            from skimage.segmentation import mark_boundaries


                            # ✅ LIME Explanation for Captured Image
                            st.subheader(translate_text("LIME Explanation", dest=lang_code))
                            temp, mask = explain_prediction_with_lime(image_arr, disease_model)
                            original_image = image_arr[0].astype('uint8')
                            fig, ax = plt.subplots()
                            ax.imshow(mark_boundaries(original_image, mask))
                            ax.axis('off')
                            st.pyplot(fig)

                            # Treatment
                            treatment = get_treatment(predicted_disease, lang_code, temperature_input, humidity_input, rainfall_input)
                            if treatment:
                                st.info(treatment)
                            else:
                                st.warning(translate_text("No specific treatment recommendation available for this disease yet.", dest=lang_code))
                        else:
                            st.error(translate_text("Failed to get prediction.", dest=lang_code))

                        import os
                        os.remove(temp_file_path)
    elif app_mode == "Crop Recommendation":
        st.title(translate_text("Crop Recommendation System", dest=lang_code))

        # Get values from session_state or fallback to 0.0
        N = st.number_input(translate_text("Nitrogen (N)", dest=lang_code), min_value=0.0, max_value=200.0,
                            value=st.session_state.get("nitrogen_input", 0.0))
        P = st.number_input(translate_text("Phosphorus (P)", dest=lang_code), min_value=0.0, max_value=200.0,
                            value=st.session_state.get("phosphorus_input", 0.0))
        K = st.number_input(translate_text("Potassium (K)", dest=lang_code), min_value=0.0, max_value=200.0,
                            value=st.session_state.get("potassium_input", 0.0))
        temperature = st.number_input(translate_text("Temperature (°C)", dest=lang_code), min_value=0.0, max_value=50.0,
                                    value=st.session_state.get("temperature_input", 0.0))
        humidity = st.number_input(translate_text("Humidity (%)", dest=lang_code), min_value=0.0, max_value=100.0,
                                value=st.session_state.get("humidity_input", 0.0))
        ph = st.number_input(translate_text("pH", dest=lang_code), min_value=0.0, max_value=14.0,
                            value=st.session_state.get("ph_input", 0.0))
        rainfall = st.number_input(translate_text("Rainfall (mm)", dest=lang_code), min_value=0.0, max_value=500.0,
                                value=st.session_state.get("rainfall_input", 0.0))

        if st.button(translate_text("Predict Crop", dest=lang_code)):
            result = predict_crop(N, P, K, temperature, humidity, ph, rainfall, lang_code)
            if result:
                st.success(result)

    elif app_mode == "Fertilizer Recommendation":
        st.title(translate_text("Fertilizer Recommendation System", dest=lang_code))

        crop_options = crop_encoder.classes_.tolist()
        crop_display, crop_mapping = get_crop_display_and_mapping(crop_options, lang_code)

        # Get default or voice-set crop
        default_crop_display = st.session_state.get("selected_crop_display", crop_display[0])

        try:
            selected_index = crop_display.index(default_crop_display)
        except ValueError:
            selected_index = 0

        # Voice-aware crop dropdown
        selected_crop_display = st.selectbox(
            translate_text("Select Crop", dest=lang_code),
            crop_display,
            index=selected_index,
            key="selected_crop_display"
        )
        selected_crop = crop_mapping[selected_crop_display]

        # Voice-aware NPK values
        N = st.number_input(
            translate_text("Nitrogen (N)", dest=lang_code),
            min_value=0,
            value=int(st.session_state.get("nitrogen_input", 0))
        )
        P = st.number_input(
            translate_text("Phosphorous (P)", dest=lang_code),
            min_value=0,
            value=int(st.session_state.get("phosphorus_input", 0))
        )
        K = st.number_input(
            translate_text("Potassium (K)", dest=lang_code),
            min_value=0,
            value=int(st.session_state.get("potassium_input", 0))
        )

        if st.button(translate_text("Recommend Fertilizer", dest=lang_code)):
            fert_result = predict_fertilizer_simple(N, P, K, selected_crop)

            fertilizer_dict_hi = {
                "10-26-26": "10-26-26 (नाइट्रोजन, फॉस्फोरस)",
                "14-35-14": "14-35-14 (उच्च फॉस्फोरस)",
                "17-17-17": "17-17-17 (संतुलित उर्वरक)",
                "20-20": "20-20 (संतुलित उर्वरक)",
                "28-28": "28-28 (ज्यादा ताकत)",
                "DAP": "डीएपी (डाय-अमोनियम फॉस्फेट)",
                "Urea": "यूरिया"
            }

            if fert_result:
                if lang_code == 'hi':
                    fert_hi = fertilizer_dict_hi.get(fert_result, fert_result)
                    st.success(f"✅ अनुशंसित उर्वरक: {fert_hi}")
                else:
                    st.success(f"✅ Recommended Fertilizer: {fert_result}")




if __name__ == "__main__":
    main()





# === Voice Input Support with Field Mapping ===
import speech_recognition as sr
import re

def recognize_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening for voice input... Please speak clearly.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("🔄 Recognizing...")
            command = recognizer.recognize_google(audio)
            st.success(f"🗣️ You said: {command}")
            return command
        except sr.UnknownValueError:
            st.warning("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
        except Exception as e:
            st.error(f"Voice input error: {e}")
    return ""

def parse_voice_input(command):
    fields = {
        "nitrogen": "nitrogen_input",
        "phosphorus": "phosphorus_input",
        "potassium": "potassium_input",
        "temperature": "temperature_input",
        "humidity": "humidity_input",
        "ph": "ph_input",
        "rainfall": "rainfall_input"
    }

    for key in fields:
        pattern = rf"{key}\s*([0-9]+(?:\.[0-9]+)?)"  # Match "key 90" or "key 90.5"
        match = re.search(pattern, command.lower())
        if match:
            value = float(match.group(1))
            st.session_state[fields[key]] = value
            st.success(f"✅ Set {key.title()} to {value}")

# Voice control trigger in sidebar
with st.sidebar:
    if st.button("🎤 Voice Command"):
        voice_command = recognize_voice_command()
        if voice_command:
            st.session_state['voice_command'] = voice_command
            parse_voice_input(voice_command)

##################
def parse_voice_input(command):
    fields = {
        "nitrogen": "nitrogen_input",
        "phosphorus": "phosphorus_input",
        "potassium": "potassium_input",
    }

    # Handle N, P, K
    for key in fields:
        pattern = rf"{key}\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, command.lower())
        if match:
            st.session_state[fields[key]] = float(match.group(1))
            st.success(f"✅ Set {key.title()} to {match.group(1)}")

    # Handle crop name from voice
    crop_pattern = r"(?:crop\s+(?:is|should be|select)?\s*)(\w+)"
    crop_match = re.search(crop_pattern, command.lower())
    if crop_match:
        spoken_crop = crop_match.group(1).strip().lower()

        crop_options = crop_encoder.classes_.tolist()
        crop_display, crop_mapping = get_crop_display_and_mapping(crop_options, "en")

        # Find and match the voice crop with the dropdown options
        for display_crop in crop_display:
            if display_crop.lower() == spoken_crop:
                st.session_state["selected_crop_display"] = display_crop
                st.success(f"🌱 Crop set to: {display_crop}")
                break
        else:
            st.warning(f"⚠️ Could not match crop '{spoken_crop}' to crop list.")



# ========================== AI Chatbot Assistant ==========================


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@st.cache_data(show_spinner=False)
def get_models():
    models = genai.list_models()  # generator
    return [m.name for m in models]  # iterate directly over it


models_list = get_models()

st.sidebar.title("🤖 Select AI Model")


def get_ai_response(user_question, model_name):
    model = genai.GenerativeModel(model_name)

    try:
        question = user_question.lower().strip()

        # Hindi and English short usage questions
        if (
            "how to use crop recommendation" in question or
            "kaise istemal kare" in question or
            "कैसे इस्तेमाल करें" in question or
            "fasal sifarish pranali ka kha jakar istemal" in question
        ):
            return "Sidebar mein jaakar 'Crop Recommendation' chunein, apne inputs dein, aur 'Predict' dabayein."

        elif (
            "how to use fertilizer recommendation" in question or
            "fertilizer recommendation ka kaise istemal kare" in question or
            "खाद सिफारिश प्रणाली का उपयोग कैसे करें" in question or
            "khad ka istemal kaise kare" in question
        ):
            return "Sidebar mein 'Fertilizer Recommendation' chunein, crop aur soil ka input dein, phir predict karein."

        elif (
            "how to use disease prediction" in question or
            "disease prediction ka kaise istemal kare" in question or
            "रोग भविष्यवाणी प्रणाली का उपयोग कैसे करें" in question or
            "rog ka pata kaise kare" in question or 
            "rog janne ka tarika kya hai" in question
        ):
            return "Sidebar mein 'Disease Prediction' section kholen, leaf image upload ya to camera se khiche aur result dekhein."

        # Everything else gets a full Gemini answer
        response = model.generate_content(user_question)
        return response.text

    except Exception as e:
        return f"❌ Error getting response: {e}"


def recognize_voice_chat():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.info("🎤 Listening for your question...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            st.sidebar.success(f"🗣️ You said: {command}")
            return command
        except sr.UnknownValueError:
            st.sidebar.warning("Could not understand your voice.")
        except sr.RequestError:
            st.sidebar.error("Speech service unavailable.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    return ""

def chatbot_ui():
    st.sidebar.header("🤖 Chat with Agri-Assistant")
    st.sidebar.markdown("Ask anything about how to use disease, crop, or fertilizer prediction.")
    # Load available models once
    models_list = get_models()
    
    # Model selector dropdown
    selected_model = st.sidebar.selectbox("Select AI Model", models_list, index=0)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.sidebar.text_input("Ask your question:", key="chat_input")

    if st.sidebar.button("🎤 Speak"):
        voice_text = recognize_voice_chat()
        if voice_text:
            user_input = voice_text
            response = get_ai_response(user_input, selected_model)
            st.session_state.chat_history.append((user_input, response))

    if st.sidebar.button("Send") and user_input:
        response = get_ai_response(user_input, selected_model)
        st.session_state.chat_history.append((user_input, response))


    # Display chat history
    for user_q, bot_r in reversed(st.session_state.chat_history):
        st.sidebar.markdown(f"**You:** {user_q}")
        st.sidebar.markdown(f"**Bot:** {bot_r}")

# === Call the chatbot inside the main app ===
chatbot_ui()

# ==========================================================================


# ================= Multilingual and Voice Support for Chatbot =================
import speech_recognition as sr
import googletrans
from googletrans import Translator

translator = Translator()

# Helper to detect language and translate to English for the model
def translate_to_english(text):
    detected_lang = translator.detect(text).lang
    if detected_lang != 'en':
        translated = translator.translate(text, src=detected_lang, dest='en').text
        return translated, detected_lang
    return text, 'en'

def translate_from_english(text, target_lang):
    if target_lang != 'en':
        return translator.translate(text, src='en', dest=target_lang).text
    return text

# # ============================================================================
