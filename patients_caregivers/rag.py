import os
import asyncio
import functools
import logging
import httpx
import pandas as pd
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent.workflow import AgentWorkflow

from llama_index.llms.openai_like import OpenAILike
from database import get_user 

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

# --- STEP 1: Configure LLM via vLLM ---
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "change-me")
EMBED_MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    "./embed_model/models--BAAI--bge-base-en-v1.5/snapshots/<snapshot-id>",
)
INDEX_PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "./index")

llm = OpenAILike(
    model="mistral-small-3.2-24b-instruct-2506",
    api_base=VLLM_API_BASE,
    api_key=VLLM_API_KEY,
    context_window=32768,
    max_tokens=2048,
    is_chat_model=True,
    is_function_calling_model=False,
)

# --- STEP 2: Configure local embedding model (BGE) ---
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_PATH,
    device=os.getenv("EMBED_DEVICE", "cpu")
)


# ---- STEP 3: Load Vector Index ----
# Load index from the local publishable index directory
storage_context_local = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
index_local = load_index_from_storage(storage_context_local)

# Note: Query engine will be created per-query with language filters

# --- Helper: Check if query is relevant ---
async def is_query_relevant(question: str, language: str) -> bool:
    prompt = (
        f"Input language is {language}, output is english. Is the following question related to dementia or palliative care. DO NOT answer the question, only answer 'yes' or 'no'."
        f"Question: {question}"
    )
    relevance_check = await llm.acomplete(prompt)
    result = relevance_check.text.strip().lower()
    logger.info(f"Relevance check result: {result}")
    return result.startswith("yes")

# --- Define an async function to handle a query ---
async def query_rag(question: str, user_id: str, answer_style: str = "", pilot_site: str = "", extended: bool = False) -> dict:
    logger.info(f"Query received: {question} | user_id: {user_id} | pilot_site: {pilot_site} | extended: {extended}")
    
    # Define language code mapping
    language_code_map = {
        "en": "english",
        "de": "german",
        "sl": "slovenian",
        "es": "spanish",
        "pt": "portuguese"
    }
    
    
    
    # Fetch user profile
    logger.info(f"Fetching user profile for user_id: {user_id}")
    user_row = get_user(user_id)
    logger.info(f"User profile fetched: {user_row}")
    

    
    # Define system directives based on user role
    system_directive_map = {
        "Patient": "Explain in simple and supportive terms suitable for patients.",
        "Family caregiver": "Use clear language with empathy for caregivers."
    }
    system_directive = "Use neutral and informative language for general audiences."
    
    if user_row:
        (
            _, age_group, gender, relationship, stage, diagnosis,
            diagnosis_other, language, device, tech_comfort, tech_savviness,
            has_support, experience, occupation, queries_completed, _,
            sus_q1, sus_q2, sus_q3, sus_q4, sus_q5,
            sus_q6, sus_q7, sus_q8, sus_q9, sus_q10,
            sus_score, sus_completed_at
        ) = user_row
        
        # Setting answer style based on user role
        if answer_style:
            # relationship = answer_style
            system_directive = system_directive_map.get(answer_style, system_directive)
        else: 
            system_directive = system_directive_map.get(relationship, system_directive)
        
        diagnosis_str = diagnosis_other if diagnosis == "Other" else diagnosis

        # Build stage description differently depending on role
        if relationship == "Family caregiver":
            stage_line = ""
            primary_diagosis_line = f"The user is caring for a person diagnosed with: {diagnosis_str}."
        elif relationship == "Patient":
            stage_line = ""
            primary_diagosis_line = f"The user has been diagnosed with: {diagnosis_str}."
        else:
            stage_line = ""
            primary_diagosis_line = ""
        
        
        user_profile_str = (
            f"Age: ({age_group}), Gender: {gender}.\n"
            f"Role: {relationship}.\n"
            # f"{stage_line}\n"
            f"{primary_diagosis_line}\n"
            f"Preferred language: {language}.\n"
        )

        language_mapped = language_code_map.get(language, "english")
        

    else:
        logger.warning(f"User ID {user_id} not found in database.")
        user_profile_str = "User profile unavailable."
        system_directive = "Use neutral and informative language for general audiences."
        language_mapped = "english"
    
    # Construct comprehensive system prompt with guidelines
    brevity_instruction = "" if extended else "- CRITICAL: Keep your response very brief - maximum 2-3 sentences. Be concise and direct.\n"
    
    system_guidelines = f"""You are a helpful AI assistant for dementia and palliative care information.

Your role is to provide supportive, accurate information about dementia, dementia care, and palliative care ONLY.

Important Guidelines:
{brevity_instruction}- Answer questions clearly and concisely related to dementia, dementia care, caregiving, and palliative care
- Only discuss topics related to dementia types (Alzheimer's, vascular dementia, Lewy body, etc.), symptoms, care strategies, caregiver support, and palliative care
- If asked about unrelated topics, politely redirect: "I can only help with questions about dementia and palliative care. Please ask about topics related to these areas."
- Do not provide medical diagnoses or specific medical advice
- Do not recommend specific treatments, medications, or dosages
- Suggest consulting healthcare professionals for medical concerns and treatment decisions
- Use clear, compassionate language appropriate for the user
- Provide emotional support and understanding for caregivers and patients
- Focus on practical information and coping strategies
- Respect the emotional challenges faced by patients and caregivers

User Context:
- Respond in Language: {language_mapped}
- Response Style: {system_directive}
- {user_profile_str}

Question: {question}"""

    personalized_query = system_guidelines
    
    # Log the full prompt
    logger.info(f"FULL PROMPT SENT TO LLM:\n{personalized_query}")
    logger.info(f"=" * 80)
    
    # Check relevance first
    logger.info("Checking query relevance...")
    if not await is_query_relevant(question, language_mapped):
        logger.info(f"Relevance check result negative")
        return {
            "answer": "This question does not appear to be about dementia or palliative care. Please ask something relevant to those topics.",
            "sources": []
        }
    logger.info(f"Relevance check result positive.")
    
    logger.info(f"Using local Mistral LLM for all users, filtering by language: {language}, pilot_site: {pilot_site}")
    
    # Create query engine with language and pilot_site filters
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    
    # Build filters list
    filters_list = [ExactMatchFilter(key="language", value=language)]
    
    # Add pilot_site filter if specified
    if pilot_site:
        filters_list.append(ExactMatchFilter(key="pilot_site", value=pilot_site))
        logger.info(f"Applied pilot_site filter: {pilot_site}")
    
    query_engine = index_local.as_query_engine(
        llm=llm,
        similarity_top_k=15,  # Increased to ensure 5+ unique sources after deduplication
        filters=MetadataFilters(filters=filters_list),
        response_mode="compact"  # Avoid multiple refining calls for faster responses
    )
    
    query_bundle = QueryBundle(personalized_query)
    logger.info(f"Query bundle: {query_bundle}")
    
    response = await query_engine.aquery(query_bundle)
    answer = response.response
    
    # Log the full answer
    logger.info(f"FULL ANSWER FROM LLM:\n{answer}")
    logger.info(f"=" * 80)
    
    sources = []
    seen_refs = set()
    for node in response.source_nodes:
        # Stop after collecting 5 unique sources
        if len(sources) >= 5:
            break
            
        metadata = node.metadata
        doc_type = metadata.get("doc_type", "unknown")
        if doc_type == "metadata":
            doc_type = "web"
        title = metadata.get("title", "Untitled")

        # Use actual web URL if available
        if "url" in metadata and metadata["url"] != "N/A":
            source_ref = metadata["url"]
        elif "source" in metadata:
            # Convert local path (e.g., "data/file.pdf") to static URL (e.g., "/static/docs/file.pdf")
            filename = os.path.basename(metadata["source"])
            source_ref = f"static/docs/{filename}"
        else:
            source_ref = "Unavailable"

        # Skip duplicates
        if source_ref in seen_refs:
            continue
        seen_refs.add(source_ref)
        
        sources.append({
            "title": title,
            "doc_type": doc_type,
            "url": source_ref,
            "source": doc_type,
            "snippet": node.text[:100]
        })
    
    
    logger.info("Finished query_rag.")
    return {
        "answer": answer,
        "sources": sources
    }


# --- Direct chat endpoint for external API access ---
async def query_education_chat(message: str, center_id: str, language: str, extended: bool = False) -> dict:
    """
    Simplified RAG query for external API access.
    Uses ./index and filters by center_id (pilot_site) and language.
    
    Args:
        message: User question
        center_id: Pilot site/center identifier for filtering
        language: Language code (en, de, sl, es, pt)
        extended: If True, provides detailed response; if False, provides brief 2-3 sentence response
    """
    logger.info(f"Education chat query - message: {message}, center_id: {center_id}, language: {language}, LLM: Mistral, extended: {extended}")
    
    # Language code mapping
    language_code_map = {
        "en": "english",
        "de": "german",
        "sl": "slovenian",
        "es": "spanish",
        "pt": "portuguese"
    }
    language_full = language_code_map.get(language, "english")
    
    # Build comprehensive system prompt with guidelines
    brevity_instruction = "" if extended else "- CRITICAL: Keep your response very brief - maximum 2-3 sentences. Be concise and direct.\n"
    
    system_prompt = f"""You are a helpful AI assistant for dementia and palliative care information.

Important Guidelines:
{brevity_instruction}- Answer questions clearly and concisely related to dementia, dementia care, caregiving, and palliative care
- Use patient-friendly language with clear and simple explanations
- Only discuss topics related to dementia types (Alzheimer's, vascular dementia, Lewy body, etc.), symptoms, care strategies, caregiver support, and palliative care
- If asked about unrelated topics, politely redirect: "I can only help with questions about dementia and palliative care. Please ask about topics related to these areas."
- Do not provide medical diagnoses or specific medical advice
- Do not recommend specific treatments, medications, or dosages
- Suggest consulting healthcare professionals for medical concerns and treatment decisions
- Use clear, compassionate language appropriate for patients and caregivers
- Provide emotional support and understanding for caregivers and patients
- Focus on practical information and coping strategies
- Respect the emotional challenges faced by patients and caregivers
- Respond in {language_full} language"""
    
    # Build filters
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    filters_list = [ExactMatchFilter(key="language", value=language)]
    
    # Valid pilot sites/centers
    valid_centers = ["INTRAS", "UCC", "UKB", "UKCM", "UoL", "UP"]
    
    if center_id and center_id in valid_centers:
        filters_list.append(ExactMatchFilter(key="pilot_site", value=center_id))
        logger.info(f"Applied center_id filter: {center_id}")
    elif center_id:
        logger.warning(f"Invalid center_id '{center_id}' - using language filter only. Valid centers: {valid_centers}")
    
    # Create query engine with selected LLM
    query_engine_local = index_local.as_query_engine(
        llm=llm,
        similarity_top_k=15,  # Increased to ensure 5+ unique sources after deduplication
        filters=MetadataFilters(filters=filters_list),
        response_mode="compact"  # Avoid multiple refining calls for faster responses
    )
    
    # Build full query with system prompt
    full_query = f"{system_prompt}\n\nQuestion: {message}"
    logger.info(f"Full query: {full_query}")
    
    # Execute query
    response = await query_engine_local.aquery(QueryBundle(full_query))
    answer = response.response
    
    # Log the full answer for education chat
    logger.info(f"EDUCATION CHAT - FULL ANSWER FROM LLM:\n{answer}")
    logger.info(f"=" * 80)
    
    # Process sources
    sources = []
    seen_refs = set()
    for node in response.source_nodes:
        # Stop after collecting 5 unique sources
        if len(sources) >= 5:
            break
            
        metadata = node.metadata
        doc_type = metadata.get("doc_type", "unknown")
        if doc_type == "metadata":
            doc_type = "web"
        title = metadata.get("title", "Untitled")
        
        # Extract pilot_site and language from metadata
        pilot_site = metadata.get("pilot_site", "Unknown")
        source_language = metadata.get("language", "Unknown")
        
        # Use actual web URL if available
        if "url" in metadata and metadata["url"] != "N/A":
            source_ref = metadata["url"]
        elif "source" in metadata:
            # For education materials, use center_id in path
            source_path = metadata["source"]
            filename = os.path.basename(source_path)
            
            # Check if it's from education-material folder
            if "education-material" in source_path and pilot_site:
                from urllib.parse import quote
                source_ref = f"https://djc-dev-ai4hope.feri.um.si/edu/djc-education-material/{pilot_site}/{quote(filename)}"
            else:
                # Fallback to static path
                source_ref = f"static/docs/{pilot_site}/{filename}"
        else:
            source_ref = "Unavailable"
        
        # Skip duplicates
        if source_ref in seen_refs:
            continue
        seen_refs.add(source_ref)
        
        sources.append({
            "title": title,
            "source": doc_type,
            "url": source_ref,
            "snippet": node.text[:100],
            "center_id": pilot_site,
            "language": source_language
        })
    
    logger.info("Finished query_education_chat.")
    return {
        "answer": answer,
        "sources": sources
    }


