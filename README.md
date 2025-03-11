# Taming LLMs

## Overview
This project implements a classification and content analysis tool using the **Groq API**, evaluating different **prompting strategies** to determine their effectiveness. The tool classifies text into predefined categories and analyzes confidence levels. A test harness was also built to compare prompt strategies, with results visualized using **Matplotlib**.

## Features
- **Groq API Integration**: Handles text classification with structured and few-shot prompting.
- **Error Handling**: Implements retry mechanisms for API rate limits and fallback methods for unsupported log probabilities.
- **Prompt Strategy Comparison**: Evaluates basic, structured, and few-shot prompting.
- **Visualization**: Uses Matplotlib to compare response times, confidence levels, and response lengths.

## Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **pip**
- **Groq API Key** (Required for model interactions)

### Setup
Clone the repository and install dependencies:
```sh
$ git clone https://github.com/yourusername/taming-llms-assignment3.git
$ cd taming-llms-assignment3
$ pip install -r requirements.txt
```

## Usage
### Running the Tool
```sh
$ python main.py --input "your_text_here"
```

### Comparing Prompt Strategies
```sh
$ python compare_strategies.py
```
This will generate a **Matplotlib** graph displaying performance differences.

## Prompt Strategies Implemented
1. **Basic Prompting**: Simple classification request.
2. **Structured Prompting**: Uses clear markers (e.g., `## Input Text`, `## Analysis`).
3. **Few-shot Prompting**: Includes multiple examples for better accuracy.

## Challenges & Solutions
- **Logprob Support Issue**: Implemented a fallback mechanism to estimate confidence levels.
- **Matplotlib Backend Error**: Resolved by switching to TkAgg.
- **API Rate Limits**: Implemented exponential backoff to handle request failures.

