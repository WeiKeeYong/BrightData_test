# Bright Data & OpenAI Research Agent

## Credits & Inspiration

A huge shout-out to NeuralNine for making the original code, and here his youtube line 

- **[Coding A Next-Gen Search Engine in Python](https://www.youtube.com/watch?v=yvXcu38rBU4)**
- Most instruction getting the keys and link to register for Bright Data is in the youtube video. 

Saw the sample crated by NeuralNine took his code, make some changes and also give me chance to figure out how to build a smart research agent. It uses the Bright Data API to pull live data from the web and OpenAI's models to think and act.

I wanted an agent that could do more than just a simple search, so I gave it a bunch of tools:
- Google & Bing Search
- Reddit Search for community discussions
- X (Twitter) Search for real-time info
- ChatGPT & Perplexity for deeper, more synthesized answers

## Three Flavors of Code

As I was building this, I experimented with a few different methods. You'll find three separate Python scripts in here, each showing a different way to build the same agent:

1.  **`BrightData-search.py`**: This one uses the popular LangChain and LangGraph libraries. It's a great example of how to use those frameworks to build powerful agents.
2.  **`BrightData-search-without-Langchains.py`**: Here, I tried building the agent from scratch, using some simple regex to understand the AI's commands. It's a look under the hood at how these agents work.
3.  **`BrightData-Search-using-OpenaAI-Tools.py`**: This version uses the official OpenAI Tools framework. It's a more modern and robust way to handle the "tool-using" part of the agent.

