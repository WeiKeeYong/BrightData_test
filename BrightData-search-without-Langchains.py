# using https://www.youtube.com/watch?v=yvXcu38rBU4 as reference and convert to use OpenAI without Langchain and dynamic tools method

from openai import OpenAI
import sys, os, time, requests, datetime
import json, re, logging
from typing import Dict, List, Callable, Any

def setup_logger():
    """Setup logger with daily file rotation and datetime formatting"""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create filename with current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(logs_dir, f"log_{current_date}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='a', encoding='utf-8'),  # Append mode
            # Uncomment the line below if you also want console output
            # logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def log_info(message):
    """Helper function to log info messages"""
    logger.info(message)

def log_error(message):
    """Helper function to log error messages"""
    logger.error(message)

def read_api_key_from_file(file_path, start_with):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith(start_with):
                    return line.strip().split(':')[1].strip()
        raise ValueError(f"Key not found starting with '{start_with}' in file '{file_path}'")
    except FileNotFoundError:
        log_error(f"File '{file_path}' not found.")
        sys.exit(1)
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)

# Initialize logger
logger = setup_logger()

Logging = True

# Define your API key for OpenAI
openai_api_key = read_api_key_from_file(r"d:\codes\keys\keys.txt", 'OPENAI:')

BRIGHTDATA_API_KEY = read_api_key_from_file(r"d:\codes\keys\keys.txt", 'BrightData:')
BRIGHTDATA_SERP_ZONE = 'serp_api01' # Change this to your desired SERP zone
BRIGHTDATA_GPT_DATASET_ID = read_api_key_from_file(r"d:\codes\keys\keys.txt", 'BrightData_chatgpt_dataset_id:')
BRIGHTDATA_PERPLEXITY_DATASET_ID = read_api_key_from_file(r"d:\codes\keys\keys.txt", 'BrightData_perplexity_dataset_id:')

if Logging:
    log_info('------New session started------')
    log_info(f'BrightData SERP Zone: {BRIGHTDATA_SERP_ZONE}')
    log_info(f'BrightData GPT Dataset ID: {BRIGHTDATA_GPT_DATASET_ID}')
    log_info(f'BrightData Perplexity Dataset ID: {BRIGHTDATA_PERPLEXITY_DATASET_ID}')

HEADERS = {
    'Authorization': f'Bearer {BRIGHTDATA_API_KEY}',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def call(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def google_search(query):
    print('Google tool is being used...')
    log_info('Google tool is being used...')

    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f'https://google.com/search?q={requests.utils.quote(query)}&brd_json=1',
        'format': 'raw',
        'country': 'US'
    }
    if Logging:
        log_info(f'Google Search Payload: {json.dumps(payload, indent=2)}')

    data = requests.post('https://api.brightdata.com/request?async=true', headers=HEADERS, json=payload).json()
    if Logging:
        log_info(f'Google Search Response: {json.dumps(data, indent=2)}')

    results = []

    for item in data.get('organic', []):
        results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item.get('description', '')}")

    if Logging:
        log_info(f'Google Search Results: {chr(10).join(results)[:10000]}')
    return '\n\n'.join(results)[:10000]

def bing_search(query):
    print('Bing tool is being used...')
    log_info('Bing tool is being used...')

    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f'https://bing.com/search?q={requests.utils.quote(query)}&brd_json=1',
        'format': 'raw',
        'country': 'US'
    }
    if Logging:
        log_info(f'Bing Search Payload: {json.dumps(payload, indent=2)}')

    data = requests.post('https://api.brightdata.com/request?async=true', headers=HEADERS, json=payload).json()
    if Logging:
        log_info(f'Bing Search Response: {json.dumps(data, indent=2)}')

    results = []

    for item in data.get('organic', []):
        results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item.get('description', '')}")

    if Logging:
        log_info(f'Bing Search Results: {chr(10).join(results)[:10000]}')

    return '\n\n'.join(results)[:10000]

def reddit_search(query):
    print('Reddit tool is being used...')
    log_info('Reddit tool is being used...')

    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f"https://google.com/search?q={requests.utils.quote('site:reddit.com ' + query)}&brd_json=1",
        'format': 'raw',
        'country': 'US'
    }
    if Logging:
        log_info(f'Reddit Search Payload: {json.dumps(payload, indent=2)}')

    data = requests.post('https://api.brightdata.com/request?async=true', headers=HEADERS, json=payload).json()
    if Logging:
        log_info(f'Reddit Search Response: {json.dumps(data, indent=2)}')    

    results = []

    for item in data.get('organic', []):
        results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item.get('description', '')}")

    if Logging:
        log_info(f'Reddit Search Results: {chr(10).join(results)[:10000]}')

    return '\n\n'.join(results)[:10000]

def x_search(query):
    print('X tool is being used...')
    log_info('X tool is being used...')

    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f"https://google.com/search?q={requests.utils.quote('site:x.com ' + query)}&brd_json=1",
        'format': 'raw',
        'country': 'US'
    }
    if Logging:
        log_info(f'X Search Payload: {json.dumps(payload, indent=2)}')

    data = requests.post('https://api.brightdata.com/request?async=true', headers=HEADERS, json=payload).json()
    if Logging:
        log_info(f'X Search Response: {json.dumps(data, indent=2)}')

    results = []

    for item in data.get('organic', []):
        results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item.get('description', '')}")

    if Logging:
        log_info(f'X Search Results: {chr(10).join(results)[:10000]}')

    return '\n\n'.join(results)[:10000]

def gpt_prompt(query):
    print('GPT tool is being used...')
    log_info('GPT tool is being used...')

    payload = [
        {
            "url": "https://chatgpt.com",
            "prompt": query
        }
    ]
    if Logging:
        log_info(f'GPT Prompt Payload: {json.dumps(payload, indent=2)}')

    url = f'https://api.brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_GPT_DATASET_ID}&format=json&custom_output_fields=answer_text_markdown'

    response = requests.post(url, headers=HEADERS, json=payload)

    snapshot_id = response.json()['snapshot_id']

    while requests.get(f'https://api.brightdata.com/datasets/v3/progress/{snapshot_id}', headers=HEADERS).json()['status'] != 'ready':
        time.sleep(5)

    data = requests.get(f'https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json', headers=HEADERS).json()[0]

    if Logging:
        log_info(f'GPT Response: {data["answer_text_markdown"]}')  
    return data['answer_text_markdown']

def perplexity_prompt(query):
    print('Perplexity tool is being used...')
    log_info('Perplexity tool is being used...')

    payload = [
        {
            "url": "https://www.perplexity.ai",
            "prompt": query
        }
    ]
    if Logging:
        log_info(f'Perplexity Prompt Payload: {json.dumps(payload, indent=2)}')

    url = f'https://api.brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_PERPLEXITY_DATASET_ID}&format=json&custom_output_fields=answer_text_markdown|sources'

    response = requests.post(url, headers=HEADERS, json=payload)

    snapshot_id = response.json()['snapshot_id']

    while requests.get(f'https://api.brightdata.com/datasets/v3/progress/{snapshot_id}', headers=HEADERS).json()['status'] != 'ready':
        time.sleep(5)

    data = requests.get(f'https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json', headers=HEADERS).json()[0]

    if Logging:
        log_info(f'Perplexity Response: {data["answer_text_markdown"]}')  
        log_info(f'Perplexity Sources: {str(data.get("sources", []))}')
    
    return data['answer_text_markdown'] + '\n\n' + str(data.get('sources', []))

# Create tool instances
tools = [
    Tool('google_search', 'Search using Google', google_search),
    Tool('bing_search', 'Search using Bing', bing_search), 
    Tool('gpt_prompt', 'Use ChatGPT to get an answer to a question', gpt_prompt),
    Tool('perplexity_prompt', 'Use Perplexity to do some research for a question', perplexity_prompt),
    Tool('reddit_search', 'Search using Reddit', reddit_search),
    Tool('x_search', 'Search using X (formally known as Twitter)', x_search)
]

def get_tool_descriptions():
    """Generate tool descriptions for the system prompt"""
    descriptions = []
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)

def parse_tool_calls(text: str) -> List[Dict]:
    """Parse tool calls from the AI response"""
    tool_calls = []
    log_info(f'Parsing tool calls from text: {text}')
    # Pattern to match tool calls like: tool_name("argument")
    pattern = r'(\w+)\s*\(\s*["\']([^"\']*)["\']?\s*\)'
    matches = re.findall(pattern, text)
    
    for tool_name, argument in matches:
        if any(tool.name == tool_name for tool in tools):
            tool_calls.append({
                'tool': tool_name,
                'argument': argument
            })
            log_info(f'Found tool call: {tool_name} with argument: {argument}')
    
    return tool_calls

def execute_tool(tool_name: str, argument: str) -> str:
    """Execute a tool by name"""
    log_info(f'Executing tool: {tool_name} with argument: {argument}')

    for tool in tools:
        if tool.name == tool_name:
            try:
                result = tool.call(argument)
                log_info(f'Tool {tool_name} executed successfully')
                return result
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                log_error(error_msg)
                return error_msg
    error_msg = f"Tool {tool_name} not found"
    log_error(error_msg)
    return error_msg

def create_system_prompt():
    """Create the system prompt with tool descriptions"""
    return f"""You are a helpful research assistant with access to multiple tools. Your goal is to provide comprehensive answers by using multiple tools to gather information.

Available tools:
{get_tool_descriptions()}

Current date and time utc: {datetime.datetime.now(datetime.timezone.utc) }

Instructions:
1. Always use at least two tools to answer questions
2. When you want to use a tool, write it in the format: tool_name("query")
3. After using tools, aggregate and summarize all the information
4. Always provide a complete list of ALL sources and links you found
5. Be thorough in your research before providing a final answer

When you want to use a tool, format it exactly like this:
google_search("your search query here")
bing_search("your search query here")
etc.

I will execute the tools for you and provide the results, then you can continue with your analysis."""

def run_agent(query: str) -> str:
    """Main agent loop that handles tool execution"""
    log_info(f'Starting agent with query: {query}')
    
    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": query}
    ]
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        log_info(f'Agent iteration {iteration + 1}/{max_iterations}')
        log_info(f'Current messages: {json.dumps(messages, indent=2)}')

        # Get response from OpenAI
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        
        ai_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_response})
        log_info(f'AI Response: {ai_response}')
        
        # Check for tool calls in the response
        tool_calls = parse_tool_calls(ai_response)
        
        if not tool_calls:
            # No more tool calls, return the final answer
            log_info('No more tool calls found. Returning final answer.')
            return ai_response
        
        log_info(f'Found {len(tool_calls)} tool calls: {[call["tool"] for call in tool_calls]}')
        
        # Execute all tool calls
        tool_results = []
        for call in tool_calls:
            result = execute_tool(call['tool'], call['argument'])
            tool_results.append(f"Result from {call['tool']}:\n{result}")
        
        # Add tool results to conversation
        tool_results_text = "\n\n".join(tool_results)
        messages.append({"role": "user", "content": f"Tool results:\n{tool_results_text}\n\nPlease continue your analysis. If you need more information, use additional tools. If you have enough information, provide your final comprehensive answer with all sources."})
        
        iteration += 1
    
    log_error("Max iterations reached")
    return "Max iterations reached. Please try with a simpler query."

if __name__ == '__main__':
    log_info('Application started')
    query = input("Query> ")
    log_info(f'User query received: {query}')
    answer = run_agent(query)
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(answer)
    log_info('Application completed')