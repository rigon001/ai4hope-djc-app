# main.py
import asyncio
import json
import logging
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from database import (
    get_connection, initialize_database, insert_user, add_interaction, get_user,
    increment_user_query_count, get_user_query_count
)
from perplexity_api import query_perplexity
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
    return templates.TemplateResponse("register.html", {"request": request})

# --- Consent endpoints ---
@app.post("/consent")
async def post_consent(request: Request, decision: str = Form(...)):
    if decision == "agree":
        request.session["consented"] = True
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
    return templates.TemplateResponse("index.html", {"request": request, "user_id": user_id, "role": role})

# --- Stream answer from query ---
@app.post("/stream")
async def stream(request: Request):
    body = await request.json()
    question = body.get("query", "")
    user_id = body.get("user_id")
    answer_style = body.get("answer_style", "")
    pilot_site = request.session.get("pilot_site", "")
    logger.info(f"Received /stream user_id: {user_id}, query: {question}, answer_style: {answer_style}, pilot_site: {pilot_site}")

    async def generate():
        prompt = (
            "You are a helpful assistant for dementia care. "
            "Speak in a patient, compassionate, and supportive way. "
            "Your answers should always be in relation to dementia or palliative care.\n\n"
            f"Answer style: {answer_style}\n"
            f"Pilot site: {pilot_site}\n"
            f"User question: {question}"
        )
        result = await query_perplexity(prompt)
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
    
    if isinstance(answer_rating, dict):
        # Professional caregiver format
        answer_rating = json.dumps(answer_rating)
    else:
        # Standard single rating
        answer_rating = int(answer_rating)
        
    if isinstance(ratings, dict):
        # Professional caregiver format
        sources = ratings.get("sources", [])
        source_ratings = json.dumps(sources)
    else:
        # Standard single rating
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
        source_ratings=source_ratings
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

@app.post("/api/perplexity/chat")
async def perplexity_chat(request: Request):
    data = await request.json()
    question = data.get("query")
    logger.info(f"In Perplexity chat.")
    # Add system prompt for dementia care context
    system_prompt = "You are a helpful assistant for dementia care. Speak in a patient, compassionate, and supportive way. Your answers should always be in relation to dementia or palliative care."
    
    # Prepend the system context to the question
    full_prompt = f"{system_prompt}\n\nUser question: {question}"
    
    logger.info(f"Perplexity chat - Original question: {question}")
    logger.info(f"Perplexity chat - Full prompt sent: {full_prompt}")
    
    response = await query_perplexity(full_prompt)
    return JSONResponse({
        "answer": response["answer"],
        "sources": response["sources"]
    })

# Temporary here should be removed for next Phase
# --- Update user language ---
@app.post("/update_language")
async def update_language(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    language = data.get("language")
    
    # Update user language in database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET language = ? WHERE user_id = ?", (language, user_id))
    conn.commit()
    conn.close()
    
    logger.info(f"Updated language for user {user_id} to {language}")
    return {"status": "updated"}

# --- Get user info ---
@app.get("/get_user_info")
async def get_user_info(user_id: str):
    user_row = get_user(user_id)
    if user_row:
        return {
            "language": user_row[7],  # language is at index 7
            "relationship": user_row[3]  # relationship is at index 3
        }
    return {"language": "en", "relationship": "Professional caregiver"}

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