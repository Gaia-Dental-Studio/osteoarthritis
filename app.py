import streamlit as st
import requests
import json

# Define the URL of the Flask API
FLASK_API_URL = "http://127.0.0.1:5000/predict"

st.title("Knee Osteoarthritis Severity Grading")

# Upload the image file
uploaded_file = st.file_uploader("Choose an image of the knee", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=400)
    
    # Submit the image to the Flask API
    if st.button("Predict Severity"):
        try:
            files = {"image": uploaded_file.getvalue()}
            response = requests.post(FLASK_API_URL, files={"image": uploaded_file})
            
            # Check if the response is valid JSON
            if response.headers.get("Content-Type") == "application/json":
                result = response.json()
                # print(result)
                
                # Display results
                st.write(f"**Predicted Severity: {result['predicted_class']} ** with confidence {result['confidence']:.2f}%")
                
                # Display confidence scores for each class
                st.write("Confidence Scores for All Classes")
                class_confidences = result["class_confidences"]
                for class_name, score in class_confidences.items():
                    st.write(f"{class_name}: {score:.2f}%")
            else:
                st.write("Error: Response from server is not JSON format.")
                st.write("Response content:", response.text)
        
        except requests.exceptions.RequestException as e:
            st.write("Error: Could not connect to the backend.")
            st.write(e)
        except ValueError as e:
            st.write("Error: Failed to decode JSON response.")
            st.write(e)
