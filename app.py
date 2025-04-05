from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pickle
import os
import tensorflow as tf
import json
from werkzeug.utils import secure_filename
import cv2
import logging
import random
import google.generativeai as genai
from deep_translator import GoogleTranslator
import requests
import json
import html
from urllib.parse import quote
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GEMINI_API_KEY ='your Gemini key'  # Replace with your real API key
genai.configure(api_key=GEMINI_API_KEY)
if not GEMINI_API_KEY:
    logger.warning("Gemini API Key not found. Some features may not work.")
    
genai.configure(api_key=GEMINI_API_KEY)



app = Flask(__name__)
chatbot_knowledge = {
    "greetings": [
        "Hello! Welcome to GreenGuard. How can I help with your farming needs today?",
        "Hi there! I'm your GreenGuard bot. Ask me about crops, diseases, or soil management.",
        "Welcome to GreenGuard! I can help you with crop recommendations, disease detection, and more."
    ],
    "crop_recommendations": [
        "Our crop recommendation system uses soil parameters like N, P, K, pH, and climate data to suggest the best crops.",
        "We can recommend suitable crops based on your soil's nutrient profile and local weather conditions.",
        "To get a crop recommendation, please provide your soil details including nitrogen, phosphorus, potassium levels, and more."
    ],
    "disease_detection": [
        "Our disease detection system can identify common plant diseases from photos of your crops.",
        "If you notice unusual spots or discoloration on your plants, upload a photo for disease detection.",
        "Early disease detection can save your harvest. Upload clear images of affected plant parts for best results."
    ],
    "farming_tips": [
        "Regular soil testing helps maintain optimal nutrient levels for your crops.",
        "Crop rotation can prevent soil nutrient depletion and reduce pest problems.",
        "Integrated pest management combines biological, cultural, and chemical controls for sustainable farming."
    ],
    "soil_health": [
        "Healthy soil should have good structure, organic matter, and a diverse microbiome.",
        "Cover crops can improve soil health by preventing erosion and adding organic matter.",
        "Maintaining proper pH levels is essential for nutrient availability to your crops."
    ],
    "default": [
        "I'm not sure I understand. Could you ask about crop recommendations, disease detection, soil health, or farming tips?",
        "I don't have information on that yet. I can help with crop recommendations, disease detection, and farming advice.",
        "That's beyond my current knowledge. Would you like information about crop recommendations or disease detection instead?"
    ]
}
agricultural_faq = {
    "crop_diseases": {
        "apple_scab": "Apple scab is a fungal disease that causes dark, scabby lesions on leaves and fruit. Treat with fungicides and improve air circulation.",
        "tomato_blight": "Tomato blight is caused by fungal pathogens. Early blight shows as dark spots with concentric rings, while late blight causes dark, water-soaked lesions. Remove infected plants and apply fungicides.",
        "powdery_mildew": "Powdery mildew appears as a white powdery coating on leaves. It thrives in high humidity and moderate temperatures. Use fungicides and improve air circulation."
    },
    "soil_nutrients": {
        "nitrogen": "Nitrogen (N) promotes leafy growth. Deficiency causes yellowing of older leaves. Sources include compost, manure, and nitrogen fertilizers.",
        "phosphorus": "Phosphorus (P) is essential for root development and flowering. Deficiency causes purplish coloration on leaves. Sources include bone meal and phosphate fertilizers.",
        "potassium": "Potassium (K) helps with overall plant health and disease resistance. Deficiency causes leaf edges to brown. Sources include wood ash and potassium-based fertilizers."
    },
    "pests": {
        "aphids": "Aphids are small sap-sucking insects that can cause leaf curling and stunted growth. Control with insecticidal soap, neem oil, or introduce natural predators like ladybugs.",
        "whiteflies": "Whiteflies are tiny white flying insects that feed on plant sap. Use yellow sticky traps, insecticidal soap, or neem oil to control infestations.",
        "caterpillars": "Caterpillars can cause extensive leaf damage. Remove by hand for small infestations or use Bacillus thuringiensis (Bt) as a biological control."
    },
    "sustainable_practices": {
        "crop_rotation": "Rotate crops annually to prevent pest buildup and nutrient depletion. Avoid planting the same family of crops in the same location for consecutive seasons.",
        "companion_planting": "Plant compatible crops together to deter pests and enhance growth. For example, plant basil near tomatoes to improve flavor and repel certain insects.",
        "water_conservation": "Use drip irrigation, collect rainwater, and mulch around plants to reduce water usage while maintaining healthy growth."
    },
    "planting_times": {
        "winter": "Winter crops include kale, spinach, carrots, and garlic in moderate climates. Start planning your spring garden during winter months.",
        "spring": "Spring is ideal for planting most vegetables. Start after the last frost with crops like peas, lettuce, and radishes, followed by warmer-season crops.",
        "summer": "Summer is good for heat-loving crops like tomatoes, peppers, and eggplants. Also plan for fall harvests by planting in mid-summer.",
        "fall": "Fall crops include leafy greens, root vegetables, and brassicas. Plant 10-12 weeks before the first expected frost."
    }
}



UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)


images_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
os.makedirs(images_dir, exist_ok=True)


crop_recommendation_model = None
model_path = os.path.join(os.path.dirname(__file__), 'models', 'RandomForest.pkl')
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            crop_recommendation_model = pickle.load(model_file)
        logger.info("Crop recommendation model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading crop recommendation model: {e}")
else:
    logger.warning(f"Crop recommendation model not found at {model_path}")


disease_model = None
tflite_interpreter = None
tflite_path = os.path.join(os.path.dirname(__file__), 'models', 'model.tflite')

logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"TFLite model path: {tflite_path}")
logger.info(f"TFLite model exists: {os.path.exists(tflite_path)}")


if os.path.exists(tflite_path):
    try:
        logger.info("Loading TFLite model")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"TFLite input details: {input_details}")
        logger.info(f"TFLite output details: {output_details}")
        
        
        def predict_with_tflite(self,img):
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            return interpreter.get_tensor(output_details[0]['index'])
        
        disease_model = type('Model', (), {'predict': predict_with_tflite})()
        logger.info("TFLite model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading TFLite model: {e}")
        disease_model = None
else:
    logger.warning(f"TFLite model not found at {tflite_path}")


idx_to_class = {}
class_indices_path = os.path.join(os.path.dirname(__file__), 'class_indices.json')
logger.info(f"Class indices path: {class_indices_path}")
logger.info(f"Class indices exists: {os.path.exists(class_indices_path)}")

if os.path.exists(class_indices_path):
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        idx_to_class = {v: k for k, v in class_indices.items()}
        logger.info("Class indices loaded successfully")
    except Exception as e:
        logger.error(f"Error loading class indices: {e}")
else:
    logger.warning("Class indices file not found. Using fallback classes for testing.")
    idx_to_class = {
        0: 'Apple___Apple_scab',
        1: 'Apple___Black_rot',
        2: 'Apple___Cedar_apple_rust',
        3: 'Apple___healthy'
    }
def enhanced_intent_recognition(user_message):
    user_message = user_message.lower()
    
    # Greeting detection
    if any(word in user_message for word in ['hello', 'hi', 'hey', 'greetings']):
        return "greetings", None
    
    # Check for specific questions in agricultural_faq
    if 'agricultural_faq' in globals():  # Ensure the variable exists
        for category, topics in agricultural_faq.items():
            for topic in topics.keys():
                if " ".join(topic.split('_')) in user_message:
                    return "specific_question", {"category": category, "topic": topic}
    
    # Category-based recognition
    category_mapping = {
        "crop_diseases": ['disease', 'infection', 'sick plant', 'spots'],
        "soil_nutrients": ['nutrient', 'fertilizer', 'npk'],
        "pests": ['pest', 'insect', 'bug'],
        "sustainable_practices": ['sustainable', 'organic', 'eco'],
        "planting_times": ['when to plant', 'planting time', 'season']
    }
    
    for category, keywords in category_mapping.items():
        if any(word in user_message for word in keywords):
            return "category_question", category
    
    # Default intent recognition fallback
    if 'recognize_intent' in globals():  # Ensure function exists
        original_intent = recognize_intent(user_message)
    else:
        original_intent = "unknown_intent"
    
    return original_intent, None


def recognize_intent(user_message):
    user_message = user_message.lower()
    
    if any(word in user_message for word in ['hello', 'hi', 'hey', 'greetings']):
        return "greetings"
    
    elif any(word in user_message for word in ['crop', 'recommend', 'suggestion', 'what to plant', 'what should i grow']):
        return "crop_recommendations"
    
    elif any(word in user_message for word in ['disease', 'sick plant', 'spot', 'infection', 'pest']):
        return "disease_detection"
    
    elif any(word in user_message for word in ['tip', 'advice', 'how to', 'best practice']):
        return "farming_tips"
    
    elif any(word in user_message for word in ['soil', 'earth', 'dirt', 'ground', 'land']):
        return "soil_health"
    
    else:
        return "default"



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
# Add these imports at the top of your app.py file


# Replace the deep_translator implementation with a direct API call to Google Translate
import requests
from urllib.parse import unquote
import logging

def translate_text(text, source_lang="auto", target_lang="en"):
    """
    Translates text using MyMemory Translation API.
    Handles various edge cases and encoding issues.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code (e.g., 'hi' for Hindi, 'auto' for auto-detect)
        target_lang (str): Target language code (e.g., 'en' for English)
        
    Returns:
        str: Translated text or original text if translation fails
    """
    try:
        # Handle URL-encoded text
        if '%' in text:
            text = unquote(text)
        
        # Log the translation attempt with sample of text
        logging.info(f"Attempting to translate from {source_lang} to {target_lang}")
        logging.debug(f"Original text (first 100 chars): {text[:100]}...")
        
        # Convert "auto" to appropriate format for MyMemory API
        langpair = f"{source_lang}|{target_lang}"
        if source_lang == "auto":
            langpair = f"|{target_lang}"
        
        # API endpoint
        url = "https://api.mymemory.translated.net/get"
        
        # Parameters
        params = {
            "q": text,
            "langpair": langpair,
            "de": "your@email.com"  # Optional: Add your email for higher usage limits
        }
        
        # Make GET request with a timeout
        response = requests.get(
            url, 
            params=params,
            timeout=15
        )
        
        # Debug log response details
        logging.debug(f"Response status code: {response.status_code}")
        logging.debug(f"Response headers: {response.headers}")
        
        # Check if the request was successful
        if response.status_code == 200:
            # Log successful response sample
            logging.debug(f"Raw response sample: {response.text[:200]}...")
            
            # Parse the response
            result = response.json()
            
            # Extract the translated text
            if result and "responseData" in result and "translatedText" in result["responseData"]:
                translated_text = result["responseData"]["translatedText"]
                
                # Check if we got a valid translation
                if translated_text and len(translated_text.strip()) > 0:
                    logging.info("Translation successful")
                    logging.debug(f"Translated text (first 100 chars): {translated_text[:100]}...")
                    return translated_text
            
            # Check for matches if responseData didn't work
            if result and "matches" in result and len(result["matches"]) > 0:
                for match in result["matches"]:
                    if "translation" in match and match["translation"]:
                        logging.info("Translation found in matches")
                        return match["translation"]
            
            logging.warning("Translation returned empty result")
            return text
        else:
            logging.error(f"Translation API returned status code: {response.status_code}")
            logging.error(f"Response content: {response.text}")
            return text
            
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

# Function for fallback to another endpoint or method
def translate_text_fallback(text, source_lang="auto", target_lang="en"):
    """
    Alternative implementation using LibreTranslate API as fallback.
    Requires a LibreTranslate instance or API key for production use.
    """
    try:
        # Handle URL-encoded text
        if '%' in text:
            text = unquote(text)
            
        logging.info(f"Attempting fallback translation from {source_lang} to {target_lang}")
        
        # Try using alternative MyMemory endpoint
        url = "https://api.mymemory.translated.net/get"
        
        # Convert "auto" to appropriate format
        if source_lang == "auto":
            # Try with empty source language
            langpair = f"|{target_lang}"
        else:
            langpair = f"{source_lang}|{target_lang}"
        
        # Parameters with different approach
        params = {
            "q": text,
            "langpair": langpair,
            "mt": "1",  # Force machine translation
            "de": "your@email.com"  # Optional: Add your email for higher usage limits
        }
        
        # Make GET request with a timeout
        response = requests.get(
            url, 
            params=params,
            timeout=15
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response
            result = response.json()
            
            # Extract the translated text
            if result and "responseData" in result and "translatedText" in result["responseData"]:
                translated_text = result["responseData"]["translatedText"]
                if translated_text and len(translated_text.strip()) > 0:
                    logging.info("Fallback translation successful")
                    return translated_text
            
            # Check for matches
            if result and "matches" in result and len(result["matches"]) > 0:
                for match in result["matches"]:
                    if "translation" in match and match["translation"]:
                        return match["translation"]
                        
            logging.warning("Fallback translation returned empty result")
            return text
        else:
            logging.error(f"Fallback translation API returned status code: {response.status_code}")
            return text
            
    except Exception as e:
        logging.error(f"Fallback translation error: {str(e)}")
        return text

# Function to detect language and handle translation with fallbacks
def robust_translate(text, target_lang="en"):
    """
    More robust translation function with fallbacks.
    Use this if you're having persistent issues with the basic functions.
    """
    # First try standard method
    translated = translate_text(text, "auto", target_lang)
    
    # If result is same as input (indicates possible failure)
    if translated == text or (translated.count('%') > 5):
        logging.warning("Primary translation failed or returned original text, trying fallback method")
        translated = translate_text_fallback(text, "auto", target_lang)
    
    # If still having issues, try with another approach
    if translated == text or (translated.count('%') > 5):
        try:
            # Try to detect language first using MyMemory
            detect_url = "https://api.mymemory.translated.net/get"
            detect_params = {
                "q": text[:100],  # Use just beginning of text for detection
                "langpair": f"|en"  # Empty source language for detection
            }
            detect_response = requests.get(detect_url, params=detect_params, timeout=5)
            if detect_response.status_code == 200:
                result = detect_response.json()
                if "responseData" in result and "detectedLanguage" in result["responseData"]:
                    detected_lang = result["responseData"]["detectedLanguage"]
                    logging.info(f"Detected language: {detected_lang}")
                    # Try again with explicit source language
                    translated = translate_text(text, detected_lang, target_lang)
        except Exception as e:
            logging.error(f"Language detection error: {str(e)}")
    
    return translated

# Update the chat route to use this new translation function
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        language = data.get('language', 'en')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"Chat message received: {user_message} in language: {language}")
        
        # Store original message
        original_message = user_message
        
        # Translate to English if not already in English
        if language != 'en':
            try:
                user_message = translate_text(user_message, source_lang=language, target_lang='en')
                logger.info(f"Translated user message: {user_message}")
            except Exception as e:
                logger.error(f"Error translating user message: {e}")
                user_message = original_message
        
        try:
            # Use Gemini for generating responses
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            prompt = f"""
            You are an AI agricultural assistant for GreenGuard. 
            Provide helpful, concise advice about farming, crops, soil health, and agricultural practices. 
            Focus on practical, actionable information. 
            Respond in a clear, informative tone suitable for a global audience.
            
            User's message: {user_message}
            """
            
            logger.info(f"Sending request to Gemini: {user_message}")
            response = model.generate_content(prompt)
            logger.info(f"Gemini response received: {len(response.text)} characters")
            
            bot_response = response.text
            if not bot_response or len(bot_response.strip()) == 0:
                logger.warning("Gemini response empty, using fallback knowledge base")
                intent, details = enhanced_intent_recognition(user_message)
                responses = chatbot_knowledge.get(intent, chatbot_knowledge["default"])
                bot_response = random.choice(responses)
            
            
            # Translate response back to original language if needed
            if language != 'en':
                try:
                    bot_response = translate_text(bot_response, source_lang='en', target_lang=language)
                    logger.info(f"Translated bot response: first 100 chars - {bot_response[:100]}...")
                except Exception as e:
                    logger.error(f"Error translating bot response: {e}")
                    # Keep English response if translation fails
            
            # Fallback to existing knowledge if Gemini fails
            if not bot_response:
                logger.warning("Empty response from Gemini, falling back to pre-defined responses")
                intent, details = enhanced_intent_recognition(user_message)
                responses = chatbot_knowledge.get(intent, chatbot_knowledge["default"])
                bot_response = random.choice(responses)
                
                # Translate fallback response if needed
                if language != 'en':
                    try:
                        bot_response = translate_text(bot_response, source_lang='en', target_lang=language)
                    except Exception as e:
                        logger.error(f"Error translating fallback response: {e}")
            
            return jsonify({"response": bot_response})
        
        except Exception as gemini_error:
            logger.error(f"Gemini API error: {gemini_error}")
            
            # Fallback to existing chatbot logic
            intent, details = enhanced_intent_recognition(user_message)
            responses = chatbot_knowledge.get(intent, chatbot_knowledge["default"])
            bot_response = random.choice(responses)

            # Translate fallback response if needed
            if language != 'en':
                try:
                    bot_response = translate_text(bot_response, source_lang='en', target_lang=language)
                except Exception as e:
                    logger.error(f"Error translating fallback response: {e}")

            if intent == "crop_recommendations":
                bot_response += " <a href='/crop-recommendation'>Try our crop recommendation tool!</a>"
            elif intent == "disease_detection":
                bot_response += " <a href='/disease-detection'>Try our disease detection tool!</a>"
            
            return jsonify({"response": bot_response})
        
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/disease-detection')
def disease_detection():
    return render_template('disease_detection.html')

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            logger.info("Received crop prediction request")
            logger.info(f"Form data: {request.form}")
            
            
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorus'])  
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            
            if crop_recommendation_model is None:
                logger.error("Crop recommendation model not loaded")
                return render_template('error.html', error="Crop recommendation model not available")

            
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)[0]
            logger.info(f"Prediction result: {prediction}")

            
            crop_details = get_crop_details(prediction)
            
            return render_template('result.html', 
                                  prediction=prediction,
                                  crop_details=crop_details,
                                  user_input={
                                      'N': N, 'P': P, 'K': K,
                                      'temperature': temperature,
                                      'humidity': humidity,
                                      'ph': ph,
                                      'rainfall': rainfall
                                  })
        except Exception as e:
            logger.error(f"Error in crop prediction: {e}")
            return render_template('error.html', error=str(e))

    
@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if request.method == 'POST':
        try:
            logger.info("Received disease detection request")
            
            
            if 'plant_image' not in request.files:
                logger.warning("No file part in request")
                return render_template('error.html', error="No file part")
            
            file = request.files['plant_image']
            
            
            if file.filename == '':
                logger.warning("No selected file")
                return render_template('error.html', error="No selected file")
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Saved uploaded file to {filepath}")
                
               
                if disease_model is None:
                    logger.error("Disease model not loaded")
                   
                    mock_disease = "Apple___Apple_scab"
                    disease_details = get_disease_details(mock_disease)
                    return render_template('disease_result.html',
                                        prediction="MOCK: " + mock_disease + " (Model not loaded)",
                                        confidence=85.5,
                                        disease_details=disease_details,
                                        image_path=f"uploads/{filename}")
                
               
                img = cv2.imread(filepath)
                if img is None:
                    logger.error(f"Failed to read image: {filepath}")
                    return render_template('error.html', error="Failed to process image")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                
                img_array = np.expand_dims(img, axis=0).astype(np.float32)
                
                prediction = disease_model.predict(img_array)
                logger.info(f"Raw prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'unknown'}")
                
                if isinstance(prediction, np.ndarray) and prediction.ndim == 2:
                    predicted_class_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_class_idx] * 100)
                else:
                    
                    predicted_class_idx = np.argmax(prediction)
                    confidence = float(prediction[predicted_class_idx] * 100)
                
                logger.info(f"Predicted class index: {predicted_class_idx}")
                
                
                disease_name = idx_to_class.get(predicted_class_idx, "Unknown")
                logger.info(f"Disease name: {disease_name}")
                
                
                disease_details = get_disease_details(disease_name)
                
                return render_template('disease_result.html',
                                      prediction=disease_name,
                                      confidence=confidence,
                                      disease_details=disease_details,
                                      image_path=f"uploads/{filename}")
            else:
                logger.warning(f"Invalid file type: {file.filename}")
                return render_template('error.html', error="File type not allowed. Please upload JPG, JPEG or PNG.")
                
        except Exception as e:
            logger.error(f"Error in disease detection: {e}")
            return render_template('error.html', error=str(e))

def get_crop_details(crop_name):
    crop_info = {
        'rice': {
            'description': 'Rice is a staple food crop for over half the world population.',
            'optimal_conditions': 'High rainfall (>200 cm), temperature 20-27°C, clayey soil with good water retention',
            'growing_season': '4-6 months depending on variety',
            'image': 'rice.jpg'
        },
        'maize': {
            'description': 'Maize (corn) is a major cereal grain grown throughout the world.',
            'optimal_conditions': 'Warm climate, well-drained soil, consistent moisture',
            'growing_season': '70-140 days depending on variety',
            'image': 'maize.jpg'
        },
        'chickpea': {
            'description': 'Chickpeas are high in protein and one of the earliest cultivated legumes.',
            'optimal_conditions': 'Cool, dry climate, well-drained soil, pH 6.0-8.0',
            'growing_season': '90-120 days',
            'image': 'chickpea.jpg'
        },
        'kidneybeans': {
            'description': 'Kidney beans are a variety of the common bean valued for their protein content.',
            'optimal_conditions': 'Warm days, cool nights, well-drained soil, pH 6.0-7.5',
            'growing_season': '90-120 days',
            'image': 'kidneybeans.jpg'
        },
        'pigeonpeas': {
            'description': 'Pigeon peas are a perennial legume crop grown in tropical regions.',
            'optimal_conditions': 'Warm climate, can tolerate drought, pH 5.0-7.0',
            'growing_season': '120-180 days',
            'image': 'pigeonpeas.jpg'
        },
        'mothbeans': {
            'description': 'Moth beans are drought-resistant legumes commonly grown in arid regions.',
            'optimal_conditions': 'Hot, dry climate, sandy loam soil',
            'growing_season': '80-90 days',
            'image': 'mothbeans.jpg'
        },
        'mungbean': {
            'description': 'Mung beans are small green beans used in both savory and sweet dishes.',
            'optimal_conditions': 'Warm climate, well-drained loamy soil, pH 6.2-7.2',
            'growing_season': '90-120 days',
            'image': 'mungbean.jpg'
        },
        'blackgram': {
            'description': 'Black gram is a bean grown in the Indian subcontinent, rich in protein.',
            'optimal_conditions': 'Warm and humid climate, loamy soil, pH 6.5-7.5',
            'growing_season': '90-120 days',
            'image': 'blackgram.jpg'
        },
        'lentil': {
            'description': 'Lentils are edible legumes high in protein, fiber, and various nutrients.',
            'optimal_conditions': 'Cool climate, well-drained soil, pH 6.0-8.0',
            'growing_season': '80-110 days',
            'image': 'lentil.jpg'
        },
        'pomegranate': {
            'description': 'Pomegranate is a fruit-bearing deciduous shrub with multiple health benefits.',
            'optimal_conditions': 'Hot, dry climate, well-drained soil, pH 5.5-7.2',
            'growing_season': 'Fruit ripens 6-7 months after flowering',
            'image': 'pomegranate.jpg'
        },'banana': {
            'description': 'Bananas are among the most popular fruits worldwide.',
            'optimal_conditions': 'Tropical climate, deep, rich, well-drained soil, pH 5.5-7.0',
            'growing_season': '9-12 months from planting to harvest',
            'image': 'banana.jpg'
        },
        'mango': {
            'description': 'Mango is a juicy stone fruit grown in tropical regions.',
            'optimal_conditions': 'Tropical climate, deep, well-drained soil, pH 5.5-7.5',
            'growing_season': 'Fruit develops 100-150 days after flowering',
            'image': 'mango.jpg'
        },
        'grapes': {
            'description': 'Grapes are a non-climacteric fruit grown on woody vines.',
            'optimal_conditions': 'Mediterranean climate, well-drained soil, pH 6.0-7.0',
            'growing_season': '150-180 days after bud break',
            'image': 'grapes.jpg'
        },
        'watermelon': {
            'description': 'Watermelon is a sweet and refreshing fruit popular in summer.',
            'optimal_conditions': 'Warm climate, sandy loam soil, pH 6.0-6.8',
            'growing_season': '80-110 days',
            'image': 'watermelon.jpg'
        },
        'muskmelon': {
            'description': 'Muskmelon (cantaloupe) is a species of melon with a sweet, aromatic flesh.',
            'optimal_conditions': 'Hot climate, well-drained sandy loam, pH 6.0-6.7',
            'growing_season': '80-120 days',
            'image': 'muskmelon.jpg'
        },
        'apple': {
            'description': 'Apples are one of the most widely cultivated tree fruits.',
            'optimal_conditions': 'Cool climate, well-drained loamy soil, pH 6.0-7.0',
            'growing_season': 'Fruit ripens 100-180 days after flowering',
            'image': 'apple.jpg'
        },
        'orange': {
            'description': 'Oranges are a popular citrus fruit known for their vitamin C content.',
            'optimal_conditions': 'Subtropical climate, well-drained soil, pH 6.0-7.5',
            'growing_season': '7-8 months from flowering to harvest',
            'image': 'orange.jpg'
        },
        'papaya': {
            'description': 'Papaya is a tropical fruit known for its sweet taste and digestive benefits.',
            'optimal_conditions': 'Tropical climate, light, well-drained soil, pH 6.0-6.5',
            'growing_season': 'First fruits 10-12 months after planting',
            'image': 'papaya.jpg'
        },
        'coconut': {
            'description': 'Coconut is a versatile tropical fruit with many culinary and non-culinary uses.',
            'optimal_conditions': 'Tropical coastal climate, sandy loam soil, high humidity',
            'growing_season': '12 months from flowering to harvest',
            'image': 'coconut.jpg'
        },
        'cotton': {
            'description': 'Cotton is a soft fiber that grows around the seeds of the cotton plant.',
            'optimal_conditions': 'Warm climate, deep, well-drained soil, pH 5.8-8.0',
            'growing_season': '150-180 days',
            'image': 'cotton.jpg'
        },
        'jute': {
            'description': 'Jute is a long, soft, shiny vegetable fiber that can be spun into strong threads.',
            'optimal_conditions': 'Warm and humid climate, loamy soil, high rainfall',
            'growing_season': '120-150 days',
            'image': 'jute.jpg'
        },
        'coffee': {
            'description': 'Coffee is a brewed drink prepared from roasted coffee beans.',
            'optimal_conditions': 'Tropical climate, rich, well-drained soil, partial shade',
            'growing_season': 'First harvest 3-4 years after planting',
            'image': 'coffee.jpg'
        }
    }
    
   
    crop_data = crop_info.get(crop_name.lower(), {
        'description': 'Information not available for this crop.',
        'optimal_conditions': 'General conditions vary by region.',
        'growing_season': 'Varies by variety and climate.',
        'image': 'default.jpg'
    })
    
    
    crop_image = crop_data['image']
    image_path = os.path.join(os.path.dirname(__file__), 'static', 'images', crop_image)
    if not os.path.exists(image_path):
        
        try:
            
            img = np.ones((400, 600, 3), dtype=np.uint8) * 240  
            cv2.putText(img, f"Crop: {crop_name}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, img)
            logger.info(f"Created placeholder image for {crop_name} at {image_path}")
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")
            crop_data['image'] = 'default.jpg'
    
    return crop_data

def get_disease_details(disease_name):
    disease_info = {
        'Apple___Apple_scab': {
            'description': 'Apple scab is a common disease of apple trees caused by the fungus Venturia inaequalis.',
            'symptoms': 'Dark olive-green to brown spots on leaves and fruit. Severe infections can cause leaf drop.',
            'treatment': 'Apply fungicides in early spring, practice good sanitation by removing fallen leaves, and select resistant varieties.',
            'prevention': 'Ensure proper spacing for air circulation, prune regularly, apply preventative fungicide sprays.'
        },
        'Apple___Black_rot': {
            'description': 'Black rot is a fungal disease caused by Botryosphaeria obtusa, affecting apples, pears and quince.',
            'symptoms': 'Circular purple spots on leaves, rotting fruit with concentric rings, and cankers on branches.',
            'treatment': 'Remove infected plant parts, apply fungicides, and maintain tree vigor.',
            'prevention': 'Prune out dead wood, remove mummified fruit, and apply protective fungicide sprays.'
        },
        'Apple___Cedar_apple_rust': {
            'description': 'Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae, requiring both apple trees and junipers to complete its life cycle.',
            'symptoms': 'Bright orange-yellow spots on leaves, deformed fruit, and orange gelatinous protrusions on infected juniper trees.',
            'treatment': 'Apply fungicides during the growing season, especially after rainy periods.',
            'prevention': 'Plant resistant apple varieties, remove nearby juniper or cedar trees if possible.'
        },
        'Apple___healthy': {
            'description': 'This plant shows no signs of disease and appears to be healthy.',
            'symptoms': 'No visible symptoms of disease. Leaves appear normal in color and shape.',
            'treatment': 'No treatment needed. Continue regular care.',
            'prevention': 'Continue good cultivation practices, including adequate watering, fertilization, and regular monitoring.'
        },
        'Corn___Common_rust': {
            'description': 'Common rust is caused by the fungus Puccinia sorghi and affects corn crops worldwide.',
            'symptoms': 'Small, circular to elongated, powdery, cinnamon-brown pustules on both leaf surfaces.',
            'treatment': 'Apply fungicides if disease appears early in the growing season.',
            'prevention': 'Plant resistant hybrids, rotate crops, and maintain good field sanitation.'
        },
        'Corn___Northern_Leaf_Blight': {
            'description': 'Northern leaf blight is a fungal disease caused by Exserohilum turcicum.',
            'symptoms': 'Long, elliptical, grayish-green or tan lesions on the leaves.',
            'treatment': 'Apply fungicides and practice crop rotation.',
            'prevention': 'Plant resistant hybrids and practice good field hygiene.'
        },
        'Corn___healthy': {
            'description': 'This corn plant shows no signs of disease and appears to be healthy.',
            'symptoms': 'No visible symptoms of disease. Leaves appear normal in color and shape.',
            'treatment': 'No treatment needed. Continue regular care.',
            'prevention': 'Maintain proper spacing, irrigation, and fertilization.'
        },
        'Tomato___Early_blight': {
            'description': 'Early blight is a fungal disease caused by Alternaria solani, affecting tomatoes and potatoes.',
            'symptoms': 'Dark brown spots with concentric rings on lower, older leaves.',
            'treatment': 'Remove infected leaves, apply copper-based fungicides.',
            'prevention': 'Mulch around plants, avoid overhead watering, ensure adequate plant spacing.'
        },
        'Tomato___Late_blight': {
            'description': 'Late blight is a devastating disease caused by the oomycete Phytophthora infestans.',
            'symptoms': 'Water-soaked pale green spots on leaves, which rapidly enlarge to become brown-black, greasy lesions.',
            'treatment': 'Apply fungicides containing copper or chlorothalonil at first signs of disease.',
            'prevention': 'Plant resistant varieties, ensure good air circulation, avoid overhead watering.'
        },
        'Tomato___Leaf_Mold': {
            'description': 'Tomato leaf mold is caused by the fungus Passalora fulva and thrives in high humidity.',
            'symptoms': 'Pale green or yellow spots on the upper leaf surface with olive-green to grayish-purple fuzzy mold on the undersides.',
            'treatment': 'Improve air circulation, reduce humidity, apply fungicides.',
            'prevention': 'Space plants properly, avoid overhead watering, use resistant varieties.'
        },
        'Tomato___Septoria_leaf_spot': {
            'description': 'Septoria leaf spot is a common disease caused by the fungus Septoria lycopersici.',
            'symptoms': 'Small, circular spots with dark borders and light centers on lower leaves.',
            'treatment': 'Remove infected leaves, apply fungicides containing chlorothalonil or copper.',
            'prevention': 'Mulch around plants, practice crop rotation, avoid overhead watering.'
        },
        'Tomato___healthy': {
            'description': 'This tomato plant shows no signs of disease and appears to be healthy.',
            'symptoms': 'No visible symptoms of disease. Leaves appear normal in color and shape.',
            'treatment': 'No treatment needed. Continue regular care.',
            'prevention': 'Maintain consistent watering, proper spacing, and adequate nutrition.'
        }
    }
    
    
    return disease_info.get(disease_name, {
        'description': 'This plant disease may require further identification.',
        'symptoms': 'Please consult with a local agricultural extension service for precise identification.',
        'treatment': 'Treatment depends on accurate disease identification.',
        'prevention': 'Good cultural practices including crop rotation, proper spacing, and sanitation.'
    })

@app.route('/about')
def about():
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500


@app.route('/system-status')
def system_status():
    status = {
        'crop_model_loaded': crop_recommendation_model is not None,
        'disease_model_loaded': disease_model is not None,
        'class_indices_loaded': len(idx_to_class) > 0,
        'upload_folder': os.path.exists(UPLOAD_FOLDER),
        'upload_folder_path': UPLOAD_FOLDER,
        'models_folder': os.path.exists(models_dir),
        'models_folder_path': models_dir,
        'tflite_model_exists': os.path.exists(tflite_path)
    }
    return render_template('status.html', status=status)

if __name__ == '__main__':
    if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
