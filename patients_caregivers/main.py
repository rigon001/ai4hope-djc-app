# main.py
import asyncio
import json
import logging
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rag import query_rag
from database import (
    get_connection, initialize_database, insert_user, add_interaction, get_user,
    increment_user_query_count, get_user_query_count
)
from datetime import datetime
from typing import List, Optional
from starlette.middleware.sessions import SessionMiddleware  # sessions for consent and user



# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),  # Log to a file
        logging.StreamHandler()          # And also to stdout
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "change-me-before-public-release")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=60 * 60 * 12)  # 12h session

# Initialize DB once
initialize_database()

# Mount static files for serving PDFs/DOCX etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates folder for HTML rendering
templates = Jinja2Templates(directory="templates")

# If you want to always create a new user on each root visit (even within the same session),
# set this to True. Otherwise, it will create once per browser session.
ALWAYS_CREATE_NEW_ON_ROOT = False

# --- Registration page --- temporary bypassed
@app.get("/", response_class=HTMLResponse)
async def register_page(request: Request):
    # Check consent first
    consented = request.session.get("consented", False)
    if not consented:
        logger.debug("Showing consent page")
        return templates.TemplateResponse("consent.html", {"request": request})
    logger.info("Displaying registration page")
    # Get selected role from session if available
    selected_role = request.session.get("selected_role", "")
    return templates.TemplateResponse("register.html", {
        "request": request,
        "selected_role": selected_role
    })

# --- Consent endpoints ---
@app.post("/consent")
async def post_consent(request: Request, decision: str = Form(...), selected_role: str = Form("")):
    if decision == "agree":
        request.session["consented"] = True
        # Store selected role in session for pre-filling registration
        if selected_role:
            request.session["selected_role"] = selected_role
            logger.info(f"User selected role from consent page: {selected_role}")
        return RedirectResponse(url="/", status_code=303)

# --- Register user and redirect to query UI ---
@app.post("/register")
async def register_user(
    request: Request,
    user_id: str = Form(""),
    age_group: str = Form(...),
    gender: str = Form(...),
    occupation: str = Form(""),
    relationship: str = Form(...),
    stage: Optional[List[str]] = Form(None),
    # care_setting: str = Form(...),
    diagnosis: str = Form(...),
    diagnosis_other: str = Form(""),
    language: str = Form(...),
    device: Optional[List[str]] = Form(None),
    tech_comfort: str = Form(...),
    tech_savviness: str = Form(...),
    has_support: str = Form(...),
    experience: str = Form(""),
    pilot_site: str = Form("")
):
    experience = int(experience) if experience.isdigit() else None
    stage_str = ", ".join(stage) if stage else ""
    device_str = ", ".join(device) if device else ""
    user_id = insert_user(
        user_id.strip() if user_id.strip() else None,
        age_group, gender,
        relationship, stage_str,
        diagnosis, diagnosis_other, language, device_str,
        tech_comfort, tech_savviness, has_support, experience,
        occupation if occupation else None
    )
    
    # Store pilot_site in session (not in database)
    request.session["pilot_site"] = pilot_site
    
    logger.info(f"User registered with ID: {user_id}, pilot_site: {pilot_site}")
    # Use a redirect status code so the browser navigates to the chat page after POST
    # 303 See Other is preferred to ensure the subsequent request is a GET
    return RedirectResponse(url=f"/chat?user_id={user_id}&role={relationship}", status_code=303)

# --- Main chat page ---
@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request, user_id: str, role: str):
    # user_id and role are now required parameters from query string
    return templates.TemplateResponse("index.html", {"request": request, "user_id": user_id, "role": role})

# --- Stream answer from query ---
@app.post("/stream")
async def stream(request: Request):
    body = await request.json()
    question = body.get("query", "")
    user_id = body.get("user_id")
    answer_style = body.get("answer_style", "")
    extended = body.get("extended", False)
    pilot_site = request.session.get("pilot_site", "")
    logger.info(f"Received /stream user_id: {user_id}, query: {question}, answer_style: {answer_style}, extended: {extended}, pilot_site: {pilot_site}")

    async def generate():
        result = await query_rag(question, user_id=user_id, answer_style=answer_style, pilot_site=pilot_site, extended=extended)
        answer = result["answer"]
        sources = result["sources"]
        logger.info(f"Streaming response for: {question}")

        # Stream the answer in chunks
        for i in range(0, len(answer), 100):
            yield answer[i:i+100]

        # Then mark the end of the answer and send sources as JSON
        yield "\n<<SOURCES>>\n"
        yield json.dumps(sources)

    return StreamingResponse(generate(), media_type="text/plain")

# --- Handle Rating and Next ---
@app.post("/rate_and_next")
async def rate_and_next(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    question = data.get("query")
    answer_style = data.get("answer_style")
    answer = data.get("answer")
    answer_rating = data.get("answer_rating")
    ratings = data.get("ratings")
    extended_clicked = data.get("extended_clicked", False)
    
    # Handle JSON rating format (new multi-question format)
    if isinstance(answer_rating, dict):
        answer_rating = json.dumps(answer_rating)
    else:
        # Legacy single rating format
        answer_rating = int(answer_rating) if answer_rating else 0
    
    source_ratings = ""

    logger.info(
        f"""Inserting to interactions table: {user_id},
            query: {question},
            answer_style: {answer_style},
            answer: {answer},
            answer_rating: {answer_rating},
            source_ratings: {source_ratings}"""
    )

    add_interaction(
        user_id=user_id,
        question=question,
        answer_style=answer_style,
        answer=answer,
        answer_rating=answer_rating,
        source_ratings=source_ratings,
        extended_clicked=extended_clicked
    )
    
    # Increment query counter
    increment_user_query_count(user_id)
    new_count = get_user_query_count(user_id)

    return {"status": "saved", "queries_completed": new_count}

# --- Get user query count ---
@app.get("/get_query_count")
async def get_query_count_endpoint(user_id: str):
    count = get_user_query_count(user_id)
    return {"queries_completed": count}

# --- External chat API for education materials ---
@app.post("/api/education-chat")
async def education_chat(request: Request):
    """
    External API endpoint for chat integration with education materials.
    Uses ./index and filters by center_id and language.
    
    Expected JSON body:
    {
        "user_id": "string",
        "message": "string",
        "center_id": "string",  # pilot_site code (e.g., "INTRAS", "UCC", etc.)
        "language": "string",   # language code (e.g., "en", "es", etc.)
        "chatbot_type": "string",  # optional, for future use
        "extended": false  # optional, default false (brief 2-3 sentences), true for detailed response
    }
    """
    from rag import query_education_chat
    
    data = await request.json()
    user_id = data.get("user_id")
    message = data.get("message")
    center_id = data.get("center_id")
    language = data.get("language", "en")
    chatbot_type = data.get("chatbot_type", "education")
    extended = data.get("extended", False)
    
    logger.info(f"Education chat API called - user_id: {user_id}, center_id: {center_id}, language: {language}, chatbot_type: {chatbot_type}, extended: {extended}")
    logger.info(f"Message: {message}")
    
    # Validate required fields
    if not message or not center_id or not language:
        return JSONResponse({
            "error": "Missing required fields: message, center_id, and language are required"
        }, status_code=400)
    
    # Query the education index (local RAG only for this study bundle)
    response = await query_education_chat(message, center_id, language, extended=extended)
    
    logger.info(f"Education chat API response generated for user {user_id}")
    return JSONResponse({
        "answer": response["answer"],
        "sources": response["sources"]
    })


# --- Save SUS responses ---
@app.post("/save_sus")
async def save_sus(request: Request):
    from database import save_sus_responses
    
    data = await request.json()
    user_id = data.get("user_id")
    sus_responses = data.get("sus_responses")
    
    if not user_id or not sus_responses:
        return JSONResponse({
            "error": "Missing required fields: user_id and sus_responses are required"
        }, status_code=400)
    
    try:
        sus_score = save_sus_responses(user_id, sus_responses)
        logger.info(f"SUS responses saved for user {user_id}, score: {sus_score}")
        return {"status": "saved", "sus_score": sus_score}
    except Exception as e:
        logger.error(f"Error saving SUS responses for user {user_id}: {e}")
        return JSONResponse({
            "error": f"Failed to save SUS responses: {str(e)}"
        }, status_code=500)
# End of temporary here