import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize OpenAI client
MODEL = 'GPT-4.1'

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("CHRISKEY")
)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=['*'],
    allow_credentials=True,
    allow_headers=['*'],
    allow_origins=['*'],
)

class ImageResponse(BaseModel):
    tumor_present: bool
    tumor_type: str
    confidence: str
    analysis: str
    recommendations: str

def encode_image(image_file):
    return base64.b64encode(image_file).decode("utf-8")
    
def generate_analysis(image_base64):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": """You are a medical imaging specialist analyzing brain scans for tumors. 
                Your response MUST begin by clearly stating:
                1. Whether a tumor is present (True/False)
                2. The type of tumor if present (or 'None' if no tumor)
                3. Your confidence level (High/Medium/Low)
                
                After this mandatory opening, provide:
                4. Detailed analysis of the scan
                5. Medical recommendations
                
                Format your response with these 5 sections separated by '|' characters"""
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze this brain scan for tumors."},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

@app.post("/analyze-brain-scan", response_model=ImageResponse)
async def analyze_brain_scan(file: UploadFile = File(...)):
    # Read the uploaded file
    image_data = await file.read()
    image_base64 = encode_image(image_data)
    
    # Get analysis from OpenAI
    full_response = generate_analysis(image_base64)
    
    # Parse the pipe-delimited response
    try:
        parts = full_response.split('|')
        if len(parts) >= 5:
            tumor_present = "true" in parts[0].lower()
            tumor_type = parts[1].strip()
            confidence = parts[2].strip()
            analysis = parts[3].strip()
            recommendations = parts[4].strip()
        else:
            raise ValueError("Unexpected response format")
    except Exception as e:
        # Fallback if parsing fails
        tumor_present = "tumor" in full_response.lower()
        tumor_type = "Unknown" if tumor_present else "None"
        confidence = "Medium"
        analysis = full_response
        recommendations = "Please consult a neurologist for professional diagnosis."
    
    # Return structured response
    return JSONResponse(content={
        "tumor_present": tumor_present,
        "tumor_type": tumor_type,
        "confidence": confidence,
        "analysis": analysis,
        "recommendations": recommendations
    })