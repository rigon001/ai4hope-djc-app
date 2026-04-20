"""Perplexity helper used by the expert-study bundle."""

import os
import httpx
import logging

PPLX_API_KEY = os.getenv("PERPLEXITY_API_KEY", "change-me")
PPLX_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
PPLX_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions")

async def query_perplexity(text: str) -> dict:
    url = PPLX_API_URL
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": PPLX_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Your answer should always be in relation to dementia or palliative care.Provide a detailed answer with sources supporting your claims."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    }

    try:
        logging.info(f"Sending query to Perplexity: {text}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            logging.info(f"Perplexity response: {result}")
            logging.info("Perplexity query successful.")
            return {
                "answer": result['choices'][0]['message']['content'],
                "sources": result.get("search_results", []),  # List of sources
            }
    except Exception as e:
        logging.error(f"Perplexity API error: {e}")
        return {
            "answer": "Sorry, we encountered an issue retrieving expert-level information.",
            "sources": []
        }
