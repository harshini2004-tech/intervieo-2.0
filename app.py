import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import InterviewPreparationModel  # Import the updated model
import requests
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
APP_ID = "55ce14f0"
APP_KEY = "b0f6219c55413f59a42a924e8286af7c"

# Configure upload settings
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize the model (replace with your actual API key)
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBbiGCBD1JmnvDeSRcLUIKOv7OgrGR6ln8')
    interview_model = InterviewPreparationModel(GOOGLE_API_KEY)
except Exception as e:
    print(f"Failed to initialize model: {e}")
    interview_model = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/api/job-details', methods=['GET'])
def get_job_details():
    # Extract parameters from the request
    job_title = request.args.get('jobTitle')
    location = request.args.get('location', '')

    if not job_title:
        return jsonify({"error": "Job title is required"}), 400

    # Adzuna API endpoint
    api_url = f"http://api.adzuna.com/v1/api/jobs/us/search/1"

    # Parameters for Adzuna API
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "what": job_title,
        "where": location,
        "content-type": "application/json",
    }

    try:
        # Make the API request
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        # Return the job results
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching job details: {e}")
        return jsonify({"error": "Failed to fetch job details"}), 500


@app.route('/api/parse-resume', methods=['POST'])
def parse_resume():
    """
    Endpoint to parse the uploaded resume
    """
    if not interview_model:
        return jsonify({"error": "Model initialization failed"}), 500

    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Create a temporary file
        temp_file = tempfile.mktemp(suffix='.pdf')
        file.save(temp_file)
        
        try:
            # Reset model state
            interview_model.reset()
            
            # Parse resume
            resume_data = interview_model.parse_resume(temp_file)
            
            # Generate interview questions
            if "error" in resume_data:
                # Fallback to generic questions if parsing fails
                questions = interview_model.generate_interview_questions()
                return jsonify({
                    "error": resume_data['error'], 
                    "questions": questions
                }), 400
            
            # Generate questions based on resume
            questions = interview_model.generate_interview_questions()
            
            return jsonify({
                "resume_data": resume_data,
                "questions": questions
            }), 200
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/evaluate-answer', methods=['POST'])
def evaluate_answer():
    """
    Endpoint to evaluate interview answer
    """
    if not interview_model:
        return jsonify({"error": "Model initialization failed"}), 500

    data = request.json
    
    if not data or 'question' not in data or 'answer' not in data:
        return jsonify({"error": "Missing question or answer"}), 400
    
    try:
        feedback = interview_model.evaluate_answer(data['question'], data['answer'])
        return jsonify({"feedback": feedback}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=5000)