import requests
from openai import OpenAI

# 每個 provider 的預設 context window 上限（tokens）
# 用於在無法動態查詢時的安全 fallback
_DEFAULT_CONTEXT_LIMITS = {
    "lmstudio": 8192,
    "ollama":   8192,
    "groq":     32768,
}

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


def _get_context_limit(provider_clean, base_url, model_name):
    """
    嘗試從 LM Studio / Ollama 動態查詢當前載入模型的 context length。
    查詢失敗時回傳 provider 的預設上限值。
    """
    try:
        if provider_clean == "lmstudio":
            res = requests.get(f"{base_url}/v1/models", timeout=3)
            if res.status_code == 200:
                for m in res.json().get("data", []):
                    if m.get("id") == model_name:
                        # LM Studio 在 model info 裡暴露 context_length
                        ctx = m.get("context_length") or m.get("max_context_length")
                        if ctx:
                            return int(ctx)
        elif provider_clean == "ollama":
            res = requests.post(f"{base_url}/api/show",
                                json={"name": model_name}, timeout=3)
            if res.status_code == 200:
                params = res.json().get("parameters", "")
                for line in str(params).splitlines():
                    if "num_ctx" in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[-1].isdigit():
                            return int(parts[-1])
    except Exception:
        pass
    return _DEFAULT_CONTEXT_LIMITS.get(provider_clean, 8192)


def _truncate_context(context: str, max_chars: int) -> str:
    """
    依字元數截斷 context，保留開頭（通常是最高 rerank 分數的段落）。
    截斷點盡量對齊段落分隔符號，避免切斷語義。
    """
    if len(context) <= max_chars:
        return context
    truncated = context[:max_chars]
    # 嘗試在最後一個段落分隔處截斷，保持語義完整
    cut = truncated.rfind("\n\n=========================================\n\n")
    if cut > max_chars // 2:
        truncated = truncated[:cut]
    truncated += "\n\n[...以下內容因模型 Context Window 上限而截斷...]"
    return truncated


def query_llm(prompt, provider="groq", model_name="", api_key="",
              custom_ip="localhost", custom_port="11434"):
    p_clean = provider.lower().strip()
    clean_ip = custom_ip.replace("http://", "").replace("https://", "").strip("/")

    if p_clean == "groq":
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        base_url = "https://api.groq.com/openai/v1"
        current_max_tokens = 1024
    elif p_clean == "lmstudio":
        base_url = f"http://{clean_ip}:{custom_port}"
        client = OpenAI(api_key="lm-studio-local", base_url=f"{base_url}/v1")
        current_max_tokens = 4096
    elif p_clean == "ollama":
        base_url = f"http://{clean_ip}:{custom_port}"
        client = OpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        current_max_tokens = 4096
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # 動態查詢模型 context limit，計算 prompt 的安全字元上限
    # 估算：1 token ≈ 1.5 中文字 / 4 英文字元，取保守值 1.5 字元/token
    # 預留 current_max_tokens 給輸出，再預留 200 tokens 給 system overhead
    if p_clean != "groq":
        ctx_limit_tokens = _get_context_limit(p_clean, base_url, model_name)
    else:
        ctx_limit_tokens = _DEFAULT_CONTEXT_LIMITS["groq"]

    safe_input_tokens = ctx_limit_tokens - current_max_tokens - 200
    safe_input_chars = max(safe_input_tokens * 2, 1000)  # 保守估算：每 token 平均 2 字元

    # 如果 prompt 超過安全字元數，截斷其中的 context 部分
    if len(prompt) > safe_input_chars:
        # 找出 context 在 prompt 中的位置並截斷
        ctx_start = prompt.find("【參考文本】：\n")
        ctx_end   = prompt.find("\n=========================================\n\n【使用者的問題】")
        if ctx_start != -1 and ctx_end != -1:
            prefix = prompt[:ctx_start + len("【參考文本】：\n")]
            suffix = prompt[ctx_end:]
            context_part = prompt[ctx_start + len("【參考文本】：\n"):ctx_end]
            # 計算 context 可以保留多少字元
            overhead_chars = len(prefix) + len(suffix)
            context_budget = safe_input_chars - overhead_chars
            if context_budget > 500:
                context_part = _truncate_context(context_part, context_budget)
                prompt = prefix + context_part + suffix
                print(f"[LLM] Context truncated to fit model context window "
                      f"({ctx_limit_tokens} tokens / ~{safe_input_chars} chars). "
                      f"Final prompt length: {len(prompt)} chars.")
            else:
                print(f"[LLM WARNING] Context budget too small ({context_budget} chars). "
                      f"Sending as-is and letting model server handle it.")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=current_max_tokens,
        temperature=0.2
    )
    return response.choices[0].message.content
