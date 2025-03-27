import os
import re
from typing import List, Dict, Any
from google import genai
from google.genai import types
import requests
from agents import Agent
from agents.agents.invocation_context import InvocationContext
from agents.models.llm_response import LlmResponse
from agents.tools import ToolContext, BaseTool
from agents.tools.agent_tool import AgentTool
import json
from pytrends.request import TrendReq
import pandas as pd
from pytrends.request import TrendReq
import pandas as pd
from typing import Dict, Any, List

# --- Helper Functions ---

def read_file(filename):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(file_path, "r") as f:
        return f.read()

def get_location_details_from_bounding_box(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> Dict[str, List[str]]:
    GOOGLE_API_KEY = "AIzaSyDh6gfu7do1pN7uQ21yPjLcvIG5Wh-t_X0"  # Replace with your API key
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    sample_points = [
        ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2),
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, min_lon),
        (max_lat, max_lon)
    ]
    zip_codes = set()
    counties = set()
    state = None
    for lat, lon in sample_points:
        params = {"latlng": f"{lat},{lon}", "key": GOOGLE_API_KEY}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                for result in data["results"]:
                    for component in result["address_components"]:
                        if "postal_code" in component["types"]:
                            zip_codes.add(component["long_name"])
                        elif "administrative_area_level_2" in component["types"]:
                            counties.add(component["long_name"])
                        elif "administrative_area_level_1" in component["types"]:
                            state = component["long_name"]
            else:
                print(f"Warning: No valid address results for lat={lat}, lon={lon}")
        else:
            print(f"Google Maps API Error: {response.status_code}, {response.text}")
            raise Exception("Failed to fetch data from Google Maps API.")
    return {
        "zip_codes": list(zip_codes) if zip_codes else ["No ZIP code found"],
        "counties": list(counties) if counties else ["No county found"],
        "state": state if state else "No state found"
    }

def google_search(query: str) -> str:
    """Performs a google search.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    client = genai.Client(
        vertexai=True,
        project="platinum-banner-303105",  # Replace with your project ID
        location="us-central1",
    )
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=query)
            ]
        )
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        tools=tools,
    )
    response = ""
    for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content.parts:
            continue
        response = response + chunk.text
    return response

# --- Tools as Functions ---

def check_area(min_lat: float, min_lon: float, max_lat: float, max_lon: float, tool_context: ToolContext) -> Dict[str, any]:
    """Retrieves ZIP codes, counties, and state names for a given bounding box.

    Args:
        min_lat (float): Minimum latitude of the bounding box.
        min_lon (float): Minimum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        tool_context: Context object that stores state data.

    Returns:
        dict: A dictionary containing:
            - "zip_codes": List of ZIP codes found in the bounding box.
            - "counties": List of counties in the bounding box.
            - "state": The state name (if consistent across results).
    """
    location_details = get_location_details_from_bounding_box(min_lat, min_lon, max_lat, max_lon)

    tool_context.state["zip_codes"] = location_details["zip_codes"]
    #tool_context.state["counties"] = location_details["counties"]
    tool_context.state["county"] = location_details["counties"][0] if location_details["counties"] else "Unknown County"
    tool_context.state["state"] = location_details["state"]
    return {
        "zip_codes": location_details["zip_codes"],
        "counties": location_details["counties"],
        "state": location_details["state"]
    }

def search_lawsuits(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches for lawsuits against infrastructure projects in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"lawsuits against infrastructure projects in {area_description}"
    results = google_search(query)
    tool_context.state["lawsuit_info"] = results
    return {"lawsuit_information": results}

def search_demographics(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches for demographic information (including affluence level) in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"demographics affluence level in {area_description}"
    results = google_search(query)
    tool_context.state["demographic_info"] = results
    return {"demographics_information": results}

def search_land_values(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches for land values in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"land values in {area_description}"
    results = google_search(query)
    tool_context.state["land_value_info"] = results
    return {"land_values_information": results}

def search_voter_demographics(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches for voter demographics in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"voter demographics in {area_description}"
    results = google_search(query)
    tool_context.state["voter_info"] = results
    return {"voter_demographics_information": results}

def search_regulations(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches for regulations related to solar energy farms in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"regulations for solar energy farms in {area_description}"
    results = google_search(query)
    tool_context.state["regulation_info"] = results
    return {"regulations_information": results}

def search_tax_breaks(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches for tax breaks for solar energy farms in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"tax breaks for solar energy farms in {area_description}"
    results = google_search(query)
    tool_context.state["tax_break_info"] = results
    return {"tax_breaks_information": results}

def search_social_media(area_description: str, tool_context: ToolContext) -> Dict[str, str]:
    """Searches social media for public sentiment about solar energy farms in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the search results.
    """
    query = f"public sentiment about solar energy farms social media in {area_description}"
    results = google_search(query)
    tool_context.state["social_media_info"] = results
    return {"social_media_information": results}

def summarize_report(tool_context: ToolContext) -> Dict[str, str]:
    """Summarizes all information into a report.

    Args:
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the report.
    """
    zip_codes = tool_context.state.get("zip_codes",)
    lawsuit_info = tool_context.state.get("lawsuit_info", {})
    demographic_info = tool_context.state.get("demographic_info", {})
    land_value_info = tool_context.state.get("land_value_info", {})
    voter_info = tool_context.state.get("voter_info", {})
    regulation_info = tool_context.state.get("regulation_info", {})
    tax_break_info = tool_context.state.get("tax_break_info", {})
    social_media_info = tool_context.state.get("social_media_info", {})
    trends_data = tool_context.state.get("trends_data", {})

    combined_info = f"""
    Zipcodes in the area: {zip_codes}
    Law Suit information: {lawsuit_info}
    Demographic information: {demographic_info}
    Land Value Information: {land_value_info}
    Voter Information:{voter_info}
    Regulation Information: {regulation_info}
    Tax Break Information: {tax_break_info}
    Social Media Information: {social_media_info}
    Google trends data:{trends_data}
    """
    query = f"summarize the below information into a report about the feasibility of a solar energyfarm, infer if the infomation is generally positive or negative. Include a table of each agent, and a score between 1-100 of how applicable the location is for a solar farm based on the information that this agent has received. {combined_info}"
    results = google_search(query)
    save_report_to_file(results)

    return {"report": results}

def save_report_to_file(tool_context: ToolContext, report_data: str):
    """Saves the report to a local text file in the 'Summaries' directory."""
    directory = "Summaries"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get location information from tool_context
    county = tool_context.state.get("county", "UnknownCounty")
    state = tool_context.state.get("state", "UnknownState")

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize county and state names for filenames
    county = re.sub(r'[^\w\-_\. ]', '_', county)
    state = re.sub(r'[^\w\-_\. ]', '_', state)

    filename = f"solar_farm_feasibility_report_{county}_{state}_{timestamp}.txt"
    filepath = os.path.join(directory, filename)

    with open(filepath, "w") as f:
        f.write(report_data)


def analyze_solar_farm_sentiment(area_description: str) -> Dict[str, Any]:
    """Analyzes public sentiment regarding solar farms in a given area using Google Trends.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).

    Returns:
        dict: A dictionary containing the analysis results.
    """

    # Initialize Google Trends API
    pytrends = TrendReq(hl='en-US', tz=360)

    # Extract location details from the input description
    print(area_description)
    location_parts = area_description.split(",")  # Assume input format: "City, State" or "County, State"

    if len(location_parts) == 2:
        county_or_city = location_parts[0].strip()
        state_abbreviation = location_parts[1].strip()
    else:
        county_or_city = area_description.strip()
        state_abbreviation = ""

    # Google Trends geo format
#    geo = f'US-{state_abbreviation}' if state_abbreviation else ''
    geo = ''
    # Define search keywords
    pro_keywords = [
        f"solar energy benefits {county_or_city}",
#        f"renewable energy {county_or_city}",
        f"solar farm incentives {county_or_city}",
#        f"clean energy jobs {county_or_city}"
    ]
    anti_keywords = [
        f"solar farm opposition {county_or_city}",
        f"solar farm environmental impact {county_or_city}",
 #       f"solar farm property values {county_or_city}",
 #       f"solar farm noise {county_or_city}",
 #       f"solar farm visual impact {county_or_city}",
 #       f"stop solar farm {county_or_city}"
    ]

    def get_trends(keywords: List[str]) -> pd.DataFrame:
        """Fetch Google Trends interest over time for given keywords."""
        try:
            pytrends.build_payload(keywords, cat=0, timeframe='today 5-y', geo=geo, gprop='')
            trends_df = pytrends.interest_over_time()
            return trends_df.to_dict()  # Convert DataFrame to dict
        except Exception as e:
            print(f"Error fetching trends: {e}")
            return {}  # Return an empty dict on error

    def get_related_queries(keywords: List[str]) -> List[str]:
        """Fetch related queries from Google Trends for given keywords."""
        related_queries = []
        for keyword in keywords:
            pytrends.build_payload([keyword], cat=0, timeframe='today 5-y', geo=geo, gprop='')
            related = pytrends.related_queries()
            if related and keyword in related and 'top' in related[keyword] and related[keyword]['top'] is not None:
                related_queries.extend([query['query'] for query in related[keyword]['top'].to_dict(orient='records')])
        return related_queries

    # Fetch sentiment trends
    pro_trends = get_trends(pro_keywords)
    anti_trends = get_trends(anti_keywords)

    # Fetch related search queries
#    related_pro_queries = get_related_queries(pro_keywords)
#    related_anti_queries = get_related_queries(anti_keywords)

    return {
        "pro_trends": pro_trends,
        "anti_trends": anti_trends,
#        "related_pro_queries": related_pro_queries,
#        "related_anti_queries": related_anti_queries,
    }


def trends_tool(area_description: str, tool_context: ToolContext):
    """Retrieves Google Trends data about solar energy farms in a given area.

    Args:
        area_description (str): A description of the area (e.g., zip code, city, state).
        tool_context: context of the tool call

    Returns:
        dict: A dictionary containing the trend data.
    """

    results = analyze_solar_farm_sentiment(area_description)
    tool_context.state["trends_data"] = results
    return results

# --- Agents ---

check_area_agent = Agent(
    model="gemini-2.0-flash-001",
    name="check_area_agent",
    description="Checks if the provided bounding box is valid and returns the zipcodes, county and state",
    tools=[check_area],
    instruction="""
You are check_area_agent. You are reponsible to take in bounding boxes and output zip codes, county and state.
Provide the zipcodes, county and State as a response.
Use the check_area tool to accomplish this task.
""",
)

trends_agent = Agent(
    model="gemini-2.0-flash-001",
    name="trends_agent",
    description="Search for public sentiment regarding solar farms in a given area using Google Trends.",
    tools=[trends_tool],
    instruction="""
    You are trends_agent. You are responsible for analyzing public sentiment regarding solar farms in a given area.
    If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
    Use the trends_tool to accomplish this task.
    You will be provided a area description, use this to search.
    """,
)

search_lawsuits_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_lawsuits_agent",
    description="Searches for lawsuits against infrastructure projects in a given area.",
    tools=[search_lawsuits],
    instruction="""
You are search_lawsuits_agent. You are responsible for gathering information about lawsuits in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_lawsuits tool to accomplish this task.
You will be provided a area description, use this to search.
""",
)

search_demographics_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_demographics_agent",
    description="Searches for demographic information (including affluence level) in a given area.",
    tools=[search_demographics],
    instruction="""
You are search_demographics_agent. You are responsible for gathering demographic information in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_demographics tool to accomplish this task.
You will be provided a area description, use this to search.
""",
)

search_land_values_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_land_values_agent",
    description="Searches for land values in a given area.",
    tools=[search_land_values],
    instruction="""
You are search_land_values_agent. You are responsible for gathering information about land values in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_land_values tool to accomplish this task.
You will be provided a area description, use this to search.
""",
)

search_voter_demographics_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_voter_demographics_agent",
    description="Searches for voter demographics in a given area.",
    tools=[search_voter_demographics],
    instruction="""
You are search_voter_demographics_agent. You are responsible for gathering information about voter demographics in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_voter_demographics tool to accomplish this task.
You will be provided a area description, use this to search.
""",
)

search_regulations_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_regulations_agent",
    description="Searches for regulations related to solar energy farms in a given area.",
    tools=[search_regulations],
    instruction="""
You are search_regulations_agent. You are responsible for gathering information about regulations related to solar energy farms in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_regulations tool to accomplish this task.
You will be provided a area description, use this to search.
""",
)

search_tax_breaks_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_tax_breaks_agent",
    description="Searches for tax breaks for solar energy farms in a given area.",
    tools=[search_tax_breaks],
    instruction="""
You are search_tax_breaks_agent. You are responsible for gathering information about tax breaks for solar energy farms in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_tax_breaks tool to accomplish this task.
You will be provided a area description, use this to search.
""",
)

search_social_media_agent = Agent(
    model="gemini-2.0-flash-001",
    name="search_social_media_agent",
    description="Searches social media for public sentiment about solar energy farms in a given area",
    tools=[search_social_media],
    instruction="""
You are search_social_media_agent. You are responsible for gathering information about public sentiment about solar energy farms in a given area.
If multiple areas are provided do not compare but consider all the information together.
    Provide a score of 1-100 of how applicable the location is for a solar farm based on the information that this agent has received.
Use the search_social_media tool to accomplish this task.
""",
)

summarize_report_agent = Agent(
    model="gemini-2.0-flash-001",
    name="summarize_report_agent",
    description="Summarizes all the gathered information into a report",
    tools=[summarize_report],
    instruction="""
    You are responsible for summarizing all the gathered information into a single report.
    Use the summarize_report tool.
    """
)

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="root_agent",
    description="Determines if a location would be a good candidate for a solar energy farm",
    flow="sequential",
    children=[check_area_agent, search_lawsuits_agent, search_demographics_agent, search_land_values_agent, search_voter_demographics_agent, search_regulations_agent, search_tax_breaks_agent, search_social_media_agent, summarize_report_agent],
    instruction="""
    You are root_agent. You are a multi agent system that determines if an area is a good candidate for a solar energy farm.
    You will need the lat long bounding boxes.
    1. Get the list of zip codes from the bounding boxes using check_area_agent.
    2. Once the list of zipcodes are available then use the tool_context.state to access the list.
    3. Then pass this to the below agents as a combined area description:
        a. use trends_agent to gather trends data for the area, search using the county and state.
        b. Use search_lawsuits_agent to find lawsuits against infrastructure in the area, search using the zipcodes
        c. Use search_demographics_agent to find demographic data for the area, search using the zipcodes
        d. Use search_land_values_agent to find the land value for the area, search using the zipcodes
        e. Use search_voter_demographics_agent to get voter information for the area, search using the zipcodes
        f. Use search_regulations_agent to gather regulation information, search using the zipcodes.
        g. Use search_tax_breaks_agent to get tax breaks for the area. Search using the zipcodes.
        h. use search_social_media_agent to gather social media information, search using the zipcodes.
    4. call summarize_report_agent to generate the final report, it will take tool_context.state as an input.
    5. Always return the report in the final output even if included in an earlier step.
    """,
)

