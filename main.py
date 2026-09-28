import os
import logging
import requests
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage



from disaster.disaster_service import get_disaster_alerts
from disaster.disaster_monitor import start_disaster_scheduler


from weather_reference import (
    UV_INDEX_REFERENCE,
    WMO_CODE_REFERENCE,
    PRECIPITATION_RANGE_REFERENCE,
    VISIBILITY_RANGE_REFERENCE,
)

from memory.conversation_memory import init_db, save_message, load_messages

load_dotenv()
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# Global variable to hold the initialized agent executor
agent_executor = None


# --- 1. Tool Definitions ---
@tool(description="Fetch Geolocation (latitude, longitude) for a given city name.")
def get_geolocation(city: str):
    """Fetch Geolocation (latitude, longitude) for a given city name."""
    url = os.environ.get("GEOLOCATION_API_EP")
    if not url:
        return "Error: GEOLOCATION_API_EP environment variable is missing."
    
    geo_url = f"{url}?name={city}&count=1"
    try:
        geo_res = requests.get(geo_url).json()
        if not geo_res.get("results"):
            return f"Could not find coordinates for {city}."
        
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        return (lat, lon)
    except Exception as e:
        return f"Error fetching coordinates: {e}"


@tool(description="Fetch the weather details for the given latitude and longitude.")
def get_weather(latitude: str, longitude: str):
    """Fetch the weather details for the given latitude and longitude."""
    url = os.environ.get("WEATHER_API_EP")
    if not url:
        return "Error: WEATHER_API_EP environment variable is missing."

    weather_url = (
        f"{url}?latitude={latitude}&longitude={longitude}"
        "&daily=weather_code,sunrise,sunset,daylight_duration,sunshine_duration,moonset,moonrise,"
        "uv_index_max,apparent_temperature_min,apparent_temperature_max,temperature_2m_min,"
        "temperature_2m_max,rain_sum&hourly=temperature_2m,weather_code,wind_speed_10m,"
        "relative_humidity_2m,precipitation,pressure_msl,soil_temperature_0cm,soil_temperature_6cm,"
        "visibility,wind_speed_80m,wind_direction_10m,wind_direction_80m,apparent_temperature,"
        "soil_temperature_18cm,uv_index,is_day,sunshine_duration&models=best_match&forecast_days=14"
    )
    return requests.get(weather_url).json()


# --- 2. FastAPI Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    
    # Environment Check
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing from environment variables or .env file.")
    
    model_name = os.environ.get("MODEL", "gemini-3.6-flash")
    
    # Initialize LLM & Agent
    llm = ChatGoogleGenerativeAI(model=model_name)
    tools = [get_geolocation, get_weather, get_disaster_alerts]

    system_prompt = SystemMessage(
        """You are a weather expert with access to 3 tools.
        Use get_geolocation() to get the geolocation for a city mentioned in the user prompt.
        Use get_weather() to get the weather details from the tool. It returns the data in a JSON format.
        Use get_disaster_alerts() to check for severe alerts, warnings, and disasters.
        If the city is not given, use the geolocation directly from the user prompt.
        The weather JSON has the data related to temperature, visibility, elevation/altitude, precipitation, uv-index, etc.
        Refer to UV_INDEX_REFERENCE, PRECIPITATION_RANGE_REFERENCE, WMO_CODE_REFERENCE, VISIBILITY_RANGE_REFERENCE.
        Identify the user persona based on the questions the user asks. 
        Never output raw JSON keys like is_day. Translate is_day: 0 to Nighttime and is_day: 1 to Daytime.
        For Example: The user persona could be a fisherman going to sea, a farmer watering crops, or an outdoor sports person going for a run or hike.
        Determine what aspect of the weather from the weather data will impact the user and advise accordingly.
        IMPORTANT INSTRUCTION: Do not use your LLM capabilities to find and interpret the weather. Use the given tools only.
        """
    )

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    


# --- 3. App Initialization & Schemas ---
    await init_db()
    
    yield


app = FastAPI(
    title="WeatherGPT",
    description="A chat service powered by LangChain and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, example="user1-session1")
    message: str = Field(..., example="can i play football outside now, i am in 12.9716,77.59?")

class ChatResponse(BaseModel):
    reply: str


# --- 4. Helper Function ---
def extract_text(content: Any) -> str:
    """Helper to extract raw text content safely from string or list-of-dicts outputs."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list) and len(content) > 0:
        if isinstance(content[0], dict) and "text" in content[0]:
            return content[0]["text"]
        return str(content[0])
    return str(content)


# --- 5. Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent is not initialized.")
    
    try:
        # Load up to the last 10 messages (5 user/assistant pairs) to prevent context explosion
        history = await load_messages(request.conversation_id, limit=25)

        # Build the message chain for this request
        messages = []
        for role, content in history:
            messages.append((role, content))

        # Append the current user request
        messages.append(("user", request.message))


        # Invoke agent with full conversational context
        response = await agent_executor.ainvoke({
            "messages": messages
        })
        
        last_message = response["messages"][-1]
        text_output = extract_text(last_message.content)
        
        # Persist the newly generated conversation turn
        await save_message(request.conversation_id, "user", request.message)
        await save_message(request.conversation_id, "assistant", text_output)
        
        return ChatResponse(reply=text_output)
    
    except Exception as e:
        logger.error(f"Error processing chat request for {request.conversation_id}: {str(e)}", exc_info=True)
        # Return generic error to user, avoiding exposing raw DB or LangChain stack traces
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred while processing your request. Please try again."
        )

@app.get("/health")
async def health_check():
    return {"status": "ok", "agent_ready": agent_executor is not None}
