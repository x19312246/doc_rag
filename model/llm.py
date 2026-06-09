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

def query_llm(prompt, provider="groq", model_name="", api_key="", custom_ip="localhost", custom_port="11434"):
    p_clean = provider.lower().strip()
    clean_ip = custom_ip.replace("http://", "").replace("https://", "").strip("/")

    if p_clean == "groq":
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        current_max_tokens = 1024
    elif p_clean == "lmstudio":
        # 不分本地遠端，完全由 custom_ip 與 custom_port 組裝連線網址
        client = OpenAI(api_key="lm-studio-local", base_url=f"http://{clean_ip}:{custom_port}/v1")
        current_max_tokens = 4096
    elif p_clean == "ollama":
        # 不分本地遠端，完全由 custom_ip 與 custom_port 組裝連線網址
        client = OpenAI(api_key="ollama", base_url=f"http://{clean_ip}:{custom_port}/v1")
        current_max_tokens = 4096
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=current_max_tokens,
        temperature=0.2
    )
    return response.choices[0].message.content