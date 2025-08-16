# Crop Recommendation and Crop Disease Prediction App

## Overview
This project is a web-based application that provides crop recommendations based on soil and environmental parameters and predicts crop diseases using image classification. The application is built using Flask, HTML, CSS, JavaScript, and Python, and it integrates a chatbot for user assistance.

## Features
- **Crop Recommendation:** Uses a Random Forest model trained on an uploaded dataset to suggest the best crop for given soil and climate conditions.
- **Disease Prediction:** Utilizes MobileNet for identifying crop diseases from uploaded images.
- **Chatbot Integration:** A Python-based chatbot to assist users with agricultural queries.
- **User-Friendly Interface:** Built with HTML, CSS, and JavaScript for an interactive experience.
- **Flask Backend:** Manages model execution, user input processing, and API handling.

## Technologies Used
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning Models:**
  - Random Forest (for crop recommendation)
  - MobileNet (for disease prediction)
- **Chatbot:** Python-based AI chatbot

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd Crop-Recommendation-Disease-Prediction
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open a browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## Dataset
- The crop recommendation model is trained on a dataset containing parameters like Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall.
- The disease prediction model uses a dataset of crop disease images for MobileNet training.

## Usage
1. **Crop Recommendation:**
   - Input soil and environmental parameters.
   - The system suggests the most suitable crop.

2. **Disease Prediction:**
   - Upload an image of a diseased crop.
   - The model predicts the disease and suggests remedies.

3. **Chatbot Assistance:**
   - Ask agricultural-related queries.
   - Get responses based on predefined logic and AI assistance.

## Future Enhancements
- Implement additional ML models for improved accuracy.
- Enhance the chatbot with NLP and deep learning.
- Integrate a real-time weather API for dynamic crop recommendations.
- Develop a mobile application version.

## Contributors
- **Kunal** (Project Developer)
- **Vaibhav Gupta** (Project Developer)

## License
This project is licensed under the MIT License.

## Acknowledgments
- Data sources for crop recommendation and disease prediction.
- Flask and MobileNet documentation for model implementation.
- Open-source tools and libraries used in this project.

Feel free to contribute or report issues in the repository!
