import os
import json
import re
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class InterviewPreparationModel:
    def __init__(self, api_key: str):
        """
        Initialize the interview preparation model with Google's Generative AI
        """
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                google_api_key=api_key
            )
            # Reset all state variables
            self.reset()
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def reset(self):
        """
        Reset the model's state
        """
        self.resume_data = None
        self.interview_questions = []
        self.current_question_index = 0

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse the uploaded resume file and extract key details
        """
        try:
            # Validate file exists and is a PDF
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            if not file_path.lower().endswith('.pdf'):
                return {"error": "Only PDF files are supported"}

            # Load the PDF resume and extract text
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split the text for processing
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            # Combine the text chunks for analysis
            resume_content = "\n".join([chunk.page_content for chunk in chunks])

            # Validate resume content
            if not resume_content.strip():
                return {"error": "Resume content is empty or could not be extracted."}

            # Prompt for detailed resume analysis
            prompt = f"""
            Extract the following details from the resume content:
            1. Skills
            2. Work experience
            3. Educational qualifications

            Resume Content:
            {resume_content}

            Please provide a detailed JSON response with:
            1. Professional skills (technical and soft skills)
            2. Work experience details
            3. Educational qualifications
            4. Notable achievements or certifications

            Strictly format as a JSON object with clear, concise information.
            """
                
            # Invoke the model
            try:
                response = self.llm.invoke(prompt)
                
                # Parse the JSON response
                try:
                    # Clean potential markdown code block
                    content = response.content
                    json_match = re.search(r'```json\n(.*?)```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                    
                    # Parse JSON
                    parsed_data = json.loads(content)
                    
                    # Store and validate parsed data
                    if not parsed_data or not isinstance(parsed_data, dict):
                        return {"error": "Invalid resume data structure"}
                    
                    self.resume_data = parsed_data
                    return parsed_data
                
                except json.JSONDecodeError as je:
                    return {
                        "error": f"Failed to parse resume data: {str(je)}",
                        "raw_response": response.content
                    }
            
            except Exception as model_error:
                return {"error": f"Model invocation failed: {str(model_error)}"}
        
        except Exception as e:
            return {"error": f"Unexpected error parsing resume: {str(e)}"}

    def generate_interview_questions(self) -> List[str]:
        """
        Generate personalized interview questions based on resume
        """
        # Fallback to generic questions if no resume data
        if not self.resume_data or "error" in self.resume_data:
            self.interview_questions = [
                "Tell me about yourself and your professional background.",
                "What are your key strengths and areas of expertise?",
                "Describe a challenging project you've worked on and how you overcame obstacles.",
                "Where do you see yourself professionally in the next 5 years?",
                "What motivates you in your career?"
            ]
            return self.interview_questions

        # Generate specific questions based on resume
        prompt = f"""
        Generate 5 personalized interview questions exploring the candidate's background:

        Skills: {self.resume_data.get('skills', [])}
        Experience: {json.dumps(self.resume_data.get('experience', []))}
        Education: {json.dumps(self.resume_data.get('education', []))}

        Questions should:
        - Be specific to the candidate's unique background
        - Cover technical and soft skill aspects
        - Encourage detailed responses
        - Reveal problem-solving capabilities
        """

        try:
            response = self.llm.invoke(prompt)
            
            # Clean and process questions
            self.interview_questions = [
                q.strip().replace('Q: ', '').replace('Question: ', '') 
                for q in response.content.split('\n') 
                if q.strip() and len(q.strip()) > 10
            ]

            # Fallback to generic questions if generation fails
            if not self.interview_questions:
                self.interview_questions = [
                    "Tell me about yourself and your professional background.",
                    "What are your key strengths and areas of expertise?",
                    "Describe a challenging project you've worked on and how you overcame obstacles.",
                    "Where do you see yourself professionally in the next 5 years?",
                    "What motivates you in your career?"
                ]

            return self.interview_questions
        
        except Exception as e:
            print(f"Error generating questions: {e}")
            self.interview_questions = [
                "Tell me about yourself and your professional background.",
                "What are your key strengths and areas of expertise?",
                "Describe a challenging project you've worked on and how you overcame obstacles.",
                "Where do you see yourself professionally in the next 5 years?",
                "What motivates you in your career?"
            ]
            return self.interview_questions

    def evaluate_answer(self, question: str, answer: str) -> str:
        """
        Provide detailed, constructive feedback on interview answers with enhanced context awareness
        """
        # Build context from resume data if available
        context = ""
        if self.resume_data:
            context = f"""
            Candidate Background:
            - Skills: {json.dumps(self.resume_data.get('skills', []))}
            - Experience Level: {len(self.resume_data.get('experience', []))} years
            - Technical Background: {json.dumps(self.resume_data.get('skills', {}).get('technical', []))}
            """

        prompt = f"""
        You are an experienced technical interviewer and career coach. Evaluate the following interview response 
        considering the candidate's background and the context of the question.

        {context}

        Question: {question}
        Candidate's Answer: {answer}

        Provide a comprehensive evaluation structured as follows:

        1. Content Analysis:
        - Key points effectively communicated
        - Technical accuracy and depth of knowledge demonstrated
        - Relevant experience and examples used
        - Alignment with industry best practices

        2. Communication Skills:
        - Clarity and structure of the response
        - Professional language and terminology usage
        - Confidence and authority in presentation
        - Balance between technical and non-technical explanation

        3. Strategic Assessment:
        - Alignment with what interviewers typically look for
        - Understanding of the underlying business/technical context
        - Problem-solving approach demonstrated
        - Strategic thinking and decision-making shown

        4. Specific Improvements:
        - Missing key points or opportunities
        - Alternative approaches or examples to consider
        - Ways to make the answer more impactful
        - Suggestions for better structuring the response

        5. Follow-up Discussion:
        - Natural follow-up questions this answer might prompt
        - Areas worth exploring further
        - Technical deep-dives that could be relevant
        - Related scenarios to demonstrate broader knowledge

        Keep feedback constructive, specific, and actionable. Focus on both immediate interview success 
        and long-term career development. If the answer involves technical concepts, evaluate both the 
        technical accuracy and the ability to communicate complex ideas effectively.
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating feedback. Please try again. Details: {str(e)}"