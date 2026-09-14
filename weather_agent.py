import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from weather_reference import (
    UV_INDEX_REFERENCE,
    WMO_CODE_REFERENCE,
    PRECIPITATION_RANGE_REFERENCE,
    VISIBILITY_RANGE_REFERENCE,
    
)

# Load environment variables from .env file
load_dotenv()

# Verify that API key exists
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from environment variables or .env file.")


# 1. Define external weather API tools
@tool(description="Fetch Geolocation (latitude, longitude) for a given city name.")
def get_geolocation(city: str):
    """Fetch Geolocation (latitude, longitude) for a given city name."""
    url=os.environ.get("GEOLOCATION_API_EP")
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
    url=os.environ.get("WEATHER_API_EP")
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


# 2. Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model=os.environ.get("MODEL") 
)

# 3. Create the agent with tool access
tools = [get_geolocation, get_weather]

system_prompt = SystemMessage(
    """You are a weather expert with access to 2 tools.
        Use get_geolocation() to get the geolocation for a city mentioned in the user prompt.
        Use get_weather() to get the weather details from the tool. It returns the data in a JSON format.
        If the city is not given, use the geolocation directly from the user prompt.
        The weather JSON has the data related to temperature, visibility, elevation/altitude, precipitation, uv-index, etc.
        Refer to UV_INDEX_REFERENCE, PRECIPITATION_RANGE_REFERENCE, WMO_CODE_REFERENCE, VISIBILITY_RANGE_REFERENCE.
        Identify the user persona based on the questions the user asks. 
        For Example: The user persona could be a fisherman going to sea, a farmer watering crops, or an outdoor sports person going for a run or hike.
        Determine what aspect of the weather from the weather data will impact the user and advise accordingly.
        IMPORTANT INSTRUCTION: Do not use your LLM capabilities to find and intrepret the weather. Use the given tools only.
        """
)

agent_executor = create_agent(
    model=llm,
    tools=tools,  # Fixed: passed 'tools' list instead of 'tool' function
    system_prompt=system_prompt
)

# 4. Query the agent
response = agent_executor.invoke({
    "messages": [("user", "can i play football outside now in Toronto, i am in 12.9716,77.59")]
})

# Print the agent's response
last_message = response["messages"][-1]
content = last_message.content

# If content is a string, print it directly
if isinstance(content, str):
    text_output = content
# If content is a list of dicts (like in your output), extract the 'text' key
elif isinstance(content, list) and len(content) > 0:
    text_output = content[0].get("text", "")
else:
    text_output = str(content)

print(text_output)