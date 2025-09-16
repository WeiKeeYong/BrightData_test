# using https://www.youtube.com/watch?v=yvXcu38rBU4 as reference, convert to use OpenAI tools framework
from openai import OpenAI
import sys, os, time, requests, datetime, json, re, logging
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

# Getting the Keys and setting from file, you can change the key retrieval method as you wish
openai_api_key = read_api_key_from_file(r"d:\codes\keys\keys.txt", 'OPENAI:')

BRIGHTDATA_API_KEY = read_api_key_from_file(r"d:\codes\keys\keys.txt", 'BrightData:')
BRIGHTDATA_SERP_ZONE = 'serp_api01' # Example zone, replace with your actual zone
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

# Tool function implementations
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

# Define tools using OpenAI's tools framework format
tools_definitions = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search using Google to find web results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use on Google"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bing_search",
            "description": "Search using Bing to find web results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use on Bing"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reddit_search",
            "description": "Search Reddit for discussions and posts",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use on Reddit"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "x_search",
            "description": "Search X (formerly Twitter) for posts and discussions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use on X"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gpt_prompt",
            "description": "Use ChatGPT to get an answer to a question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or prompt to send to ChatGPT"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perplexity_prompt",
            "description": "Use Perplexity to do research for a question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question to send to Perplexity"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Map function names to actual functions
available_functions = {
    "google_search": google_search,
    "bing_search": bing_search,
    "reddit_search": reddit_search,
    "x_search": x_search,
    "gpt_prompt": gpt_prompt,
    "perplexity_prompt": perplexity_prompt
}

def execute_function_call(function_call):
    """Execute a function call and return the result"""
    function_name = function_call.function.name
    function_args = json.loads(function_call.function.arguments)
    
    log_info(f'Executing function: {function_name} with args: {function_args}')
    
    if function_name in available_functions:
        try:
            # Get the query parameter from arguments
            query = function_args.get('query', '')
            result = available_functions[function_name](query)
            log_info(f'Function {function_name} executed successfully')
            return result
        except Exception as e:
            error_msg = f"Error executing {function_name}: {str(e)}"
            log_error(error_msg)
            return error_msg
    else:
        error_msg = f"Function {function_name} not found"
        log_error(error_msg)
        return error_msg

def create_system_prompt():
#trying different system prompts

    '''
    return f"""You are a helpful research assistant with access to multiple tools. 
                Your goal is to provide comprehensive answers by using multiple tools to gather information.

                Current date and time UTC: {datetime.datetime.now(datetime.timezone.utc)}

                Instructions:
                1. Always use at least two tools to answer questions thoroughly
                2. Use the available tools to search for information from different sources
                3. After gathering information from multiple sources, provide a comprehensive analysis
                4. Always include all sources and links you found in your final answer
                5. Be thorough in your research before providing conclusions

                Available tools include Google search, Bing search, Reddit search, X search, ChatGPT prompting, and Perplexity research."""
    '''
    return f"""You are a research assistant with access to multiple tools.  
                Your job is to answer questions comprehensively, using information from diverse and trustworthy sources.  

                Current UTC datetime: {datetime.datetime.now(datetime.timezone.utc)}

                Guidelines:
                1. Always call at least two different tools before giving your final answer.  
                2. Prioritize search tools first (Google, Bing, Reddit, X) to collect factual or up-to-date information.  
                3. Use GPT or Perplexity tools for synthesis, comparison, or filling gaps — never as the only source.  
                4. Clearly indicate which tools were used and include all links or sources found.  
                5. Provide a final answer that is concise, well-structured, and supported by the evidence you gathered.  
                6. If tools return little or no useful info, explain this clearly and suggest the best next steps."""

def run_agent(query: str) -> str:
    """Main agent loop using OpenAI's tools framework"""
    log_info(f'Starting agent with query: {query}')
    
    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": query}
    ]
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        log_info(f'Agent iteration {iteration + 1}/{max_iterations}')
        
        # Get response from OpenAI with tools
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            tools=tools_definitions,
            tool_choice="auto",
            temperature=0
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        log_info(f'AI Response: {response_message.content}')
        
        # Check if the model wants to call any tools
        if response_message.tool_calls:
            log_info(f'Found {len(response_message.tool_calls)} tool calls')
            
            # Execute each tool call
            for tool_call in response_message.tool_calls:
                log_info(f'Executing tool call: {tool_call.function.name}')
                
                function_result = execute_function_call(tool_call)
                
                # Add the tool result to messages
                tool_message = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": function_result
                }
                messages.append(tool_message)
            
            iteration += 1
        else:
            # No tool calls, return the final response
            log_info('No tool calls found. Returning final answer.')
            return response_message.content or "No response generated."
    
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