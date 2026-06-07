import requests
from openai import OpenAI

def get_local_models(base_url, provider="LM Studio"):
    try:
        if provider == "Ollama":
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        else:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [m["id"] for m in models]
    except Exception:
        pass
    return []

def query_llm(prompt, provider="Groq", model_name="", api_key="", custom_ip="localhost", custom_port="11434"):
    if provider == "Groq":
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    elif provider == "LM Studio 本地端":
        client = OpenAI(api_key="lm-studio-local", base_url="http://localhost:1234/v1")
    elif provider == "Ollama 遠端/本地":
        clean_ip = custom_ip.replace("http://", "").replace("https://", "").strip("/")
        client = OpenAI(api_key="ollama", base_url=f"http://{clean_ip}:{custom_port}/v1")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
    #current_max_tokens = 1024 if provider == "Groq" else 2560
    # 🌟 修正：將本地端與遠端的 max_tokens 放大到 4096，確保長表格重塑不會斷頭去尾
    if provider in ["LM Studio 本地端", "Ollama 遠端/本地"]:
        current_max_tokens = 4096
    else:
        current_max_tokens = 2048 if "llama3" in model_name.lower() else 1024
        
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=current_max_tokens,
        temperature=0.2
    )
    return response.choices[0].message.content