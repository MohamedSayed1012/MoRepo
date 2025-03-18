# Conversational Agent with Groq Integration

This project implements a conversational agent using Groq's API for language model capabilities. The agent can use tools to gather information from external sources and solve problems through step-by-step reasoning techniques.

## Agent Types

1. **Basic Agent**: Uses weather tools to provide weather information
2. **Chain of Thought Agent**: Adds reasoning capabilities and calculator functionality
3. **ReAct Agent**: Implements the Reasoning + Acting pattern with additional search capabilities

## Setup Instructions

### Prerequisites
- Python 3.8+
- Groq API key
- WeatherAPI key

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/conversational-agent-groq.git
cd conversational-agent-groq
```

2. Install required packages:
```bash
pip install groq python-dotenv requests
```

3. Create a `.env` file in the project root with the following variables:
```
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama3-70b-8192
WEATHER_API_KEY=your_weather_api_key_here
```

#### Creating a .env file via Command Prompt (Windows)

1. Open Command Prompt by pressing `Win + R`, typing `cmd`, and hitting Enter
2. Navigate to your project directory: `cd path\to\your\project`
3. Create an empty .env file: `type nul > .env`
4. Open the file in notepad: `notepad .env`
5. Add your environment variables and save the file

#### Creating a .env file via Terminal (macOS/Linux)

1. Open Terminal
2. Navigate to your project directory: `cd /path/to/your/project`
3. Create an empty .env file: `touch .env`
4. Edit the file: `nano .env` (or use your preferred text editor)
5. Add your environment variables and save the file

### Running the Application

Run the application with:
```bash
python conversational_agent.py
```

You will be prompted to choose which type of agent to use:
1. Basic Weather Assistant
2. Chain of Thought Agent
3. ReAct Reasoning Agent
4. Compare All Agents (Bonus feature)

## Implementation Details

### Groq Integration
The code uses the Groq Python SDK to interact with Groq's LLM API endpoints. By default, it uses the Llama 3 70B model, but you can change this in the .env file by modifying the `LLM_MODEL` variable.

### Weather Tools
- `get_current_weather`: Retrieves current weather conditions for any location
- `get_weather_forecast`: Gets a multi-day forecast for a specified location

### Calculator Tool
A simple calculator function that can evaluate mathematical expressions.

### Search Tool
A simulated web search tool that returns information based on keyword matching.

### Reasoning Patterns
1. **Basic**: Simple query-response interaction with tool use
2. **Chain of Thought**: Step-by-step reasoning with intermediate calculations
3. **ReAct**: Explicit reasoning and acting cycles with thoughts, actions, and observations

## Comparative Evaluation System
The bonus feature allows you to compare responses from all three agent types side-by-side, rate them, and save the results to a CSV file for analysis.

## Important Notes

- The tool implementations are functional but simplified for demonstration purposes
- Error handling is basic and may need enhancement for production use
- The web search tool is a simulation and returns predefined results
- The calculator tool uses Python's `eval()` function, which should be used with caution in production environments

## Example Conversations

### Basic Agent
```
You: What's the weather like in New York?
