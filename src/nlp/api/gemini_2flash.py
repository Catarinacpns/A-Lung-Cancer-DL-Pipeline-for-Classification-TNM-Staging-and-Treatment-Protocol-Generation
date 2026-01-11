import google.generativeai as genai

def generate_response_gemini(context, query):
    """Generates a response using Gemini 2.0 Flash based on structured TNM staging prompt."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY"))

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Query: {query}\n\n{context}")
    
    return response.text
