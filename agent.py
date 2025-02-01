# File: deepseek_assistant.py
import os
import json
import requests
from typing import List, Dict, Generator
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.llm.base import LLM

load_dotenv()

class DeepSeekLLM(LLM):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.temperature = 0.3
        self.max_tokens = 1000
        self.history_file = "chat_history.json"

    def get_response(self, messages: List[Dict]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def response_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        with requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    if chunk.startswith("data: "):
                        json_str = chunk[6:]
                        if json_str.strip() != "[DONE]":
                            try:
                                json_data = json.loads(json_str)
                                content = json_data["choices"][0]["delta"].get("content", "")
                                yield content
                            except json.JSONDecodeError:
                                pass

class DeepSeekAssistant:
    def __init__(self, user_id: str = "default_user"):
        self.llm = DeepSeekLLM()
        self.assistant = Assistant(
            llm=self.llm,
            system_prompt="You're a helpful AI assistant trained by DeepSeek. Provide clear, concise answers.",
            user_id=user_id,
        )
        self.history_file = f"chat_history_{user_id}.json"

    def chat(self, query: str, stream: bool = False) -> str:
        if stream:
            return self.assistant.run(query, stream=True)
        return self.assistant.run(query)

    def save_history(self):
        """Save conversation history to JSON file"""
        history = self.get_history()
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def load_history(self):
        """Load conversation history from JSON file"""
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
                self.assistant.memory.clear()
                for message in history:
                    self.assistant.memory.add(message)
        except FileNotFoundError:
            pass

    def get_history(self) -> List[dict]:
        """Get conversation history"""
        return self.assistant.memory.get()

    def clear_history(self):
        """Clear conversation history"""
        self.assistant.memory.clear()
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

def main():
    assistant = DeepSeekAssistant(user_id="demo_user")
    assistant.load_history()
    
    print("\n🌟 DeepSeek AI Assistant 🌟")
    print("Type 'exit' to quit\n")
    
    try:
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ("exit", "quit"):
                print("\nAssistant: Goodbye! Have a great day!")
                assistant.save_history()
                break
                
            if not user_input.strip():
                continue
                
            print("\nAssistant: ", end="")
            response_stream = assistant.chat(user_input, stream=True)
            full_response = []
            for chunk in response_stream:
                print(chunk, end="", flush=True)
                full_response.append(chunk)
            print("\n")
            
    except KeyboardInterrupt:
        print("\n\nAssistant: Session interrupted. Saving conversation...")
        assistant.save_history()

if __name__ == "__main__":
    main()