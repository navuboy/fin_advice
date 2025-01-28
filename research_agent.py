import json
import requests
from typing import Generator, List, Dict
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

class DeepSeekClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")  # Get from .env
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _handle_response(self, response: requests.Response):
        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")
        return response.json()
    
    def chat(self, prompt: str, stream: bool = False) -> Generator[str, None, None]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "deepseek-chat",
            "stream": stream
        }
        
        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            stream=stream
        )
        
        if stream:
            for line in response.iter_lines():
                if line:
                    line_decoded = line.decode('utf-8').strip()
                    if line_decoded.startswith("data: "):
                        try:
                            chunk = json.loads(line_decoded.split('data: ')[1])
                            yield chunk['choices'][0]['delta'].get('content', '')
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}. Line: {line_decoded}")
                    else:
                        print(f"Skipped line: {line_decoded}")
        else:
            data = self._handle_response(response)
            yield data['choices'][0]['message']['content']

class ResearchAgent:
    def __init__(self):
        self.client = DeepSeekClient()

    def generate_subtopics(self, main_topic: str) -> List[str]:
        prompt = f"""Break down the following main topic into 5 comprehensive subtopics:
        Main Topic: {main_topic}
        List each subtopic on a separate line starting with a hyphen (-).
        Do not use any markdown formatting or additional text.
        Example format:
        - Subtopic 1
        - Subtopic 2
        - Subtopic 3"""

        response = next(self.client.chat(prompt))
        
        # Parse the text response
        subtopics = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                subtopics.append(line[2:].strip())
        
        return subtopics[:5]  # Take first 5 items

    def research_subtopic(self, subtopic: str) -> str:
        
        prompt = f"""Conduct in-depth research on the following subtopic:
        Subtopic: {subtopic}
        Provide detailed analysis covering:
        - Key concepts and definitions
        - Current trends and developments - provide online links to relevant news
        - Major challenges or controversies
        - Future implications
        - Relevant case studies/examples
        
        Structure your response in Markdown format."""
        
        return next(self.client.chat(prompt))