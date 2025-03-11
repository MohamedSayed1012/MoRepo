import os
import time

from typing import Dict, List, Any, Tuple

# Set a compatible matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from dotenv import load_dotenv
import groq

#1): Configuration and Basic Completion

class LLMClient:
    def __init__(self):
        """Initialize the Groq API client with proper error handling."""
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = groq.Client(api_key=self.api_key)
        self.model = "llama3-70b-8192"  # Default model
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def complete(self, prompt: str, max_tokens: int = 1000,
                 temperature: float = 0.7, stream: bool = False,
                 logprobs: bool = False, top_logprobs: int = 5) -> Any:
        """
        Send a completion request to the Groq API.

        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            stream: Whether to stream the response
            logprobs: Whether to return log probabilities
            top_logprobs: Number of most likely tokens to return probabilities for

        Returns:
            The completion response or None if an error occurred
        """
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs if logprobs else None
                )

                if stream:
                    return response  # Return the stream object
                else:
                    return response.choices[0].message.content

            except groq.RateLimitError:
                # Handle rate limiting
                retry_count += 1
                wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                print(f"Rate limited. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            except groq.APIError as e:
                # Handle API errors
                error_message = str(e)
                # Check if error is due to unsupported logprobs
                if "logprobs" in error_message:
                    print("Logprobs not supported with this model. Falling back to regular completion.")
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=stream
                        )
                        if stream:
                            return response
                        else:
                            return response.choices[0].message.content
                    except Exception as e2:
                        print(f"Fallback completion error: {e2}")
                        return None
                if "500" in error_message or "503" in error_message:  # Server errors
                    retry_count += 1
                    print(f"Server error. Retrying {retry_count}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Unrecoverable API error: {e}")
                    return None

            except Exception as e:
                # Handle other errors
                print(f"Unexpected error: {e}")
                return None

        print(f"Failed after {self.max_retries} retries")
        return None

    def complete_with_logprobs(self, prompt: str, max_tokens: int = 1000,
                               temperature: float = 0) -> Tuple[str, Any]:
        """
        Send a completion request and return both the completion and log probabilities.

        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)

        Returns:
            Tuple of (completion_text, raw_response)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=5
            )

            completion = response.choices[0].message.content
            return completion, response

        except Exception as e:
            # If error is due to unsupported logprobs, fall back to a normal completion and use a default confidence value.
            error_message = str(e)
            if "logprobs" in error_message:
                print("Logprobs not supported. Using fallback completion without logprobs.")
                completion = self.complete(prompt, max_tokens=max_tokens, temperature=temperature)
                return completion, None  # No logprobs available
            print(f"Error in complete_with_logprobs: {e}")
            return None, None



#2): Structured Completions


def create_structured_prompt(text: str, question: str) -> str:
    """
    Creates a structured prompt that will produce a completion with
    easily recognizable sections.

    Args:
        text: The input text to analyze
        question: The question to answer about the text

    Returns:
        A structured prompt with clear section markers
    """
    prompt = f"""
# Analysis Report
## Input Text
{text}

## Question
{question}

## Analysis
"""
    return prompt


def extract_section(completion: str, section_start: str, section_end: str = None) -> str:
    """
    Extracts content between section_start and section_end.
    If section_end is None, extracts until the end of the completion.

    Args:
        completion: The text to extract from
        section_start: The starting marker
        section_end: The ending marker (optional)

    Returns:
        The extracted section or None if not found
    """
    start_idx = completion.find(section_start)
    if start_idx == -1:
        return None

    start_idx += len(section_start)

    if section_end is None:
        return completion[start_idx:].strip()

    end_idx = completion.find(section_end, start_idx)
    if end_idx == -1:
        return completion[start_idx:].strip()

    return completion[start_idx:end_idx].strip()


def stream_until_marker(client: LLMClient, prompt: str, stop_marker: str, max_tokens: int = 1000) -> str:
    """
    Streams the completion and stops once a marker is detected.
    Returns the accumulated text up to the marker.

    Args:
        client: The LLMClient instance
        prompt: The prompt to send
        stop_marker: The marker to stop at
        max_tokens: Maximum tokens to generate

    Returns:
        The accumulated text up to the marker
    """
    accumulated_text = ""
    stream = client.complete(prompt, max_tokens=max_tokens, stream=True)

    try:
        for chunk in stream:
            if not hasattr(chunk.choices[0], 'delta') or not hasattr(chunk.choices[0].delta, 'content'):
                continue

            content = chunk.choices[0].delta.content
            if content:
                accumulated_text += content

                # Check if stop marker is in the accumulated text
                if stop_marker in accumulated_text:
                    # Truncate at the stop marker
                    stop_idx = accumulated_text.find(stop_marker)
                    return accumulated_text[:stop_idx]
    except Exception as e:
        print(f"Error during streaming: {e}")

    return accumulated_text



#3): Classification with Confidence

def classify_with_confidence(client: LLMClient, text: str, categories: List[str],
                             confidence_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Classifies text into one of the provided categories.
    Returns the classification only if confidence is above threshold.

    Args:
        client: The LLMClient instance
        text: The text to classify
        categories: List of possible categories
        confidence_threshold: Minimum confidence to accept classification

    Returns:
        Dictionary with classification results
    """
    # Create a prompt that encourages clear, unambiguous classification
    prompt = f"""
Classify the following text into exactly one of these categories: {', '.join(categories)}.

Response format:
1. CATEGORY: [one of: {', '.join(categories)}]
2. CONFIDENCE: [high|medium|low]
3. REASONING: [explanation]

Text to classify:
{text}
"""
    # Get completion with logprobs (fallback if logprobs not supported)
    completion, response = client.complete_with_logprobs(prompt, max_tokens=500, temperature=0)

    if not completion:
        return {
            "category": "error",
            "confidence": 0.0,
            "reasoning": "Failed to get classification"
        }

    # Extract classification
    category = extract_section(completion, "1. CATEGORY: ", "\n")
    confidence_text = extract_section(completion, "2. CONFIDENCE: ", "\n")
    reasoning = extract_section(completion, "3. REASONING: ")

    # Convert text confidence to score
    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
    confidence_score = confidence_map.get(confidence_text.lower() if confidence_text else "", 0.5)

    # If logprobs analysis is available, you could refine confidence further.
    # Otherwise, use the default confidence_score.
    if response:
        logprob_confidence = analyze_logprobs_confidence(response, category, categories)
    else:
        logprob_confidence = confidence_score

    # Use the lower of the two confidence metrics (being conservative)
    final_confidence = min(confidence_score, logprob_confidence)

    # Return classification if confidence exceeds threshold
    if final_confidence > confidence_threshold:
        return {
            "category": category,
            "confidence": final_confidence,
            "reasoning": reasoning
        }
    else:
        return {
            "category": "uncertain",
            "confidence": final_confidence,
            "reasoning": "Confidence below threshold"
        }


def analyze_logprobs_confidence(response: Any, category: str, categories: List[str]) -> float:
    """
    Analyze log probabilities to determine model's confidence in its classification.

    Args:
        response: The raw response from the API containing logprobs
        category: The predicted category
        categories: List of all possible categories

    Returns:
        A confidence score between 0 and 1
    """
    try:
        # Placeholder for demonstration - in a real implementation
        # you would extract actual logprobs from the response and compute a confidence score
        confidence = 0.85
        return confidence

    except Exception as e:
        print(f"Error analyzing logprobs: {e}")
        return 0.7  # Default moderate confidence



#4): Prompt Strategy Comparison

def compare_prompt_strategies(client: LLMClient, texts: List[str], categories: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compares different prompt strategies on the same classification tasks.

    Args:
        client: The LLMClient instance
        texts: List of texts to classify
        categories: List of possible categories

    Returns:
        Dictionary of results for each strategy
    """
    strategies = {
        "basic": lambda text: f"Classify this text into one of these categories: {', '.join(categories)}\n\nText: {text}\nCategory:",
        "structured": lambda text: f"""
Classification Task
Categories: {', '.join(categories)}
Text: {text}
Classification:""",
        "few_shot": lambda text: f"""
Here are some examples of text classification:

Example 1:
Text: "The product arrived damaged and customer service was unhelpful."
Classification: Negative

Example 2:
Text: "While delivery was slow, the quality exceeded my expectations."
Classification: Mixed

Example 3:
Text: "Absolutely love this! Best purchase I've made all year."
Classification: Positive

Now classify this text:
Text: "{text}"
Classification:"""
    }

    results = {}
    for strategy_name, prompt_func in strategies.items():
        print(f"\nEvaluating {strategy_name} strategy...")
        strategy_results = []

        for i, text in enumerate(texts):
            print(f"  Processing text {i + 1}/{len(texts)}")
            prompt = prompt_func(text)

            # Measure time
            start_time = time.time()
            completion, response = client.complete_with_logprobs(prompt, max_tokens=100, temperature=0)
            elapsed_time = time.time() - start_time

            # Extract the classification (assuming it's the first word after the prompt)
            classification = completion.strip().split()[0] if completion else "error"

            # Calculate confidence using logprobs if available, else fallback
            confidence = 0.8  # Default placeholder confidence
            if response:
                confidence = 0.8  # In practice, refine this using logprobs

            result = {
                "text": text,
                "classification": classification,
                "confidence": confidence,
                "response_length": len(completion) if completion else 0,
                "time": elapsed_time
            }

            strategy_results.append(result)

        results[strategy_name] = strategy_results

    # Print summary statistics
    print("\nStrategy Comparison Summary:")
    for strategy, strategy_results in results.items():
        avg_time = sum(r["time"] for r in strategy_results) / len(strategy_results)
        avg_confidence = sum(r["confidence"] for r in strategy_results) / len(strategy_results)
        avg_length = sum(r["response_length"] for r in strategy_results) / len(strategy_results)
        print(f"{strategy}:")
        print(f"  Avg. Time: {avg_time:.3f}s")
        print(f"  Avg. Confidence: {avg_confidence:.3f}")
        print(f"  Avg. Response Length: {avg_length:.1f} chars")

    # Visualize results
    visualize_comparison_results(results)

    return results


def visualize_comparison_results(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Visualizes the comparison results.

    Args:
        results: Results dictionary from compare_prompt_strategies
    """
    strategies = list(results.keys())

    # Calculate metrics
    avg_times = []
    avg_confidences = []
    avg_lengths = []

    for strategy in strategies:
        strategy_results = results[strategy]
        avg_times.append(sum(r["time"] for r in strategy_results) / len(strategy_results))
        avg_confidences.append(sum(r["confidence"] for r in strategy_results) / len(strategy_results))
        avg_lengths.append(sum(r["response_length"] for r in strategy_results) / len(strategy_results))

    # Create visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Time comparison
    axs[0].bar(strategies, avg_times)
    axs[0].set_title('Average Response Time')
    axs[0].set_ylabel('Time (seconds)')

    # Confidence comparison
    axs[1].bar(strategies, avg_confidences)
    axs[1].set_title('Average Confidence')
    axs[1].set_ylabel('Confidence Score')
    axs[1].set_ylim(0, 1)

    # Length comparison
    axs[2].bar(strategies, avg_lengths)
    axs[2].set_title('Average Response Length')
    axs[2].set_ylabel('Characters')

    plt.tight_layout()
    plt.savefig('prompt_strategy_comparison.png')
    plt.show()


def main():
    """
    Main function to demonstrate the functionality of the LLM taming tool.
    """
    # Initialize the client
    try:
        client = LLMClient()
        print("Successfully initialized LLM client")
    except ValueError as e:
        print(f"Error initializing client: {e}")
        return

    # Part 1: Basic completion demo
    print("\n=== Part 1: Basic Completion ===")
    prompt = "What are three ways to improve prompt engineering?"
    print(f"Prompt: {prompt}")
    completion = client.complete(prompt, max_tokens=200)
    if completion:
        print(f"Completion: {completion}")
    else:
        print("Failed to get completion")

    # Part 2: Structured completion demo
    print("\n=== Part 2: Structured Completion ===")
    text = "The new smartphone has excellent battery life, but the camera quality is disappointing for the price point."
    question = "What are the positive and negative aspects mentioned in this review?"

    structured_prompt = create_structured_prompt(text, question)
    print("Structured prompt created")

    completion = client.complete(structured_prompt, max_tokens=300)
    if completion:
        analysis = extract_section(completion, "## Analysis\n")
        print(f"Extracted analysis: {analysis}")
    else:
        print("Failed to get structured completion")

    # Stream until marker demo
    print("\n=== Streaming Until Marker ===")
    stream_prompt = create_structured_prompt(
        "Amazon's customer service responded quickly to my inquiry, but they were unable to resolve my issue.",
        "What sentiment is expressed in this text?"
    )
    print("Streaming until '## Conclusion' marker...")
    streamed_text = stream_until_marker(client, stream_prompt, "## Conclusion", max_tokens=300)
    print(f"Streamed text (truncated at marker):\n{streamed_text}")

    # Part 3: Classification with confidence demo
    print("\n=== Part 3: Classification with Confidence ===")
    texts = [
        "This product exceeded all my expectations. Highly recommended!",
        "The quality is okay, but it's overpriced for what you get.",
        "Complete waste of money. Broke after two days of use."
    ]

    categories = ["Positive", "Neutral", "Negative"]

    for i, text in enumerate(texts):
        print(f"\nText {i + 1}: {text}")
        result = classify_with_confidence(client, text, categories, confidence_threshold=0.7)
        print(f"Classification: {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}")

    # Part 4: Prompt strategy comparison demo
    print("\n=== Part 4: Prompt Strategy Comparison ===")
    sample_texts = [
        "The customer service rep was incredibly helpful and solved my problem quickly.",
        "The product works as expected, nothing extraordinary but does the job.",
        "Terrible experience. Will never buy from this company again.",
        "While the design is nice, the functionality leaves much to be desired.",
        "I've had this for a year now and it still works perfectly."
    ]

    results = compare_prompt_strategies(client, sample_texts, categories)
    print("Prompt strategy comparison completed and visualized")



if __name__ == "__main__":
    main()
