
from diskcache import Cache
cache = Cache('./cache/gpt.cache')
from config.configurator import configs
from openai import OpenAI, PermissionDeniedError, APIError, RateLimitError, APIConnectionError, APITimeoutError
import time

GENERATIVE_MODELAS = ["gpt-3.5-turbo-instruct","gpt-4-turbo-preview" ]

class OpenAIClient:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    max_tokens: int = 4096

    OPENROUTER_BASE_URL: str = ""

    history: list = []

    use_cache: bool = configs['llm']['use_cache'] if 'use_cache' in configs['llm'] else False
    # cache_key: str = configs['llm']['cache_key'] if 'cache_key' in configs['llm'] else 'default_cache_key'

    api_keys: list = [""]  # 存储所有可用API密钥
    current_api_index: int = 0
    max_retries_per_api: int = 2
    global_max_retries: int = None

    def __init__(self, model_name:str=None, temperature:float=None, max_tokens:int=None, api_key: str = None,base_url: str = None,api_keys: list = None,):
        from openai import OpenAI
        #self.client = OpenAI(base_url=self.OPENROUTER_BASE_URL or base_url,api_key=api_key,timeout=60.0)
        self.base_url = base_url or self.OPENROUTER_BASE_URL
        if model_name: self.model_name = model_name
        if temperature: self.temperature = temperature
        if max_tokens: self.max_tokens = max_tokens
        self.is_generative_model = self.model_name in GENERATIVE_MODELAS

        self.api_keys = api_keys or self.api_keys
        if not self.api_keys:
            raise ValueError("必须提供至少一个API密钥（api_keys）")
        self.global_max_retries = self.global_max_retries or len(self.api_keys) * self.max_retries_per_api
        self.client = self._create_client()

    def _create_client(self) -> OpenAI:
        current_api_key = self.api_keys[self.current_api_index]
        return OpenAI(
            base_url=self.base_url,
            api_key=current_api_key,
            timeout=60.0
        )

    def _switch_next_api(self) -> None:
        self.current_api_index = (self.current_api_index + 1) % len(self.api_keys)
        self.client = self._create_client()
        print(f"切换到第{self.current_api_index + 1}个API密钥")

    def query_chat(self, text, stop=None, temperature=None) -> str:
        retry_count = 0
        while retry_count < self.global_max_retries:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model=self.model_name,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                    stop=stop
                )

                if not chat_completion:
                    raise ValueError("API返回空对象（chat_completion is None）")
                if not hasattr(chat_completion, "choices") or len(chat_completion.choices) == 0:
                    raise IndexError("API返回结果中choices为空，无有效内容")

                return chat_completion.choices[0].message.content

            except (
                    PermissionDeniedError,
                    APIError,
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    TypeError,
                    IndexError,
                    ValueError
            ) as e:
                print(f"API调用失败（第{retry_count + 1}次重试）：{str(e)}")
                retry_count += 1
                if retry_count % self.max_retries_per_api == 0:
                    self._switch_next_api()
                time.sleep(1)

        print(f"所有API尝试失败，无法处理请求：{text}")
        return ""

    def query_generative(self, text, stop=None, temperature=None) -> str:
        retry_count = 0
        while retry_count < self.global_max_retries:
            try:
                completion = self.client.completions.create(
                    prompt=text,
                    model=self.model_name,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                    stop=stop
                )
                return completion.choices[0].text.strip()
            except (PermissionDeniedError, APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                print(f"API调用失败（第{retry_count + 1}次重试）：{str(e)}")
                retry_count += 1
                if retry_count % self.max_retries_per_api == 0:
                    self._switch_next_api()
                time.sleep(1)
        print(f"所有API尝试失败，无法处理请求：{text}")
        return None

    def query_one(self, text, stop=None, temperature=None) -> str:
        cache_key_ = f"{self.model_name}_{text}"
        if self.use_cache and cache_key_ in cache:
            return cache.get(cache_key_)

        if self.is_generative_model:
            res = self.query_generative(text, stop=stop, temperature=temperature)
        else:
            res = self.query_chat(text, stop=stop, temperature=temperature)

        if res is not None and self.use_cache:
            cache.set(cache_key_, res)
        return res

    def query_one_stream(self, text) -> None:
        retry_count = 0
        while retry_count < self.global_max_retries:
            try:
                stream = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model=self.model_name,
                    temperature=self.temperature,
                    stream=True,
                    stop=["\nObservation:"]
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                return
            except (PermissionDeniedError, APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                print(f"\n流式调用失败（第{retry_count + 1}次重试）：{str(e)}")
                retry_count += 1
                if retry_count % self.max_retries_per_api == 0:
                    self._switch_next_api()
                time.sleep(1)
        print(f"\n所有API尝试失败，无法处理流式请求：{text}")

    def chat(self, text, stop=None, temperature=None) -> str:
        self.history.append({"role": "user", "content": text})
        retry_count = 0
        while retry_count < self.global_max_retries:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=self.history,
                    model=self.model_name,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                    stop=stop
                )
                res = chat_completion.choices[0].message.content
                self.history.append({"role": "assistant", "content": res})
                return res
            except (PermissionDeniedError, APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                print(f"对话调用失败（第{retry_count + 1}次重试）：{str(e)}")
                retry_count += 1
                if retry_count % self.max_retries_per_api == 0:
                    self._switch_next_api()
                time.sleep(1)
        print(f"所有API尝试失败，无法处理对话：{text}")
        return None

    def clear_history(self):
        self.history = []

    def chat_with_history(self, history: list, stop=None, temperature=None) -> str:
        retry_count = 0
        while retry_count < self.global_max_retries:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=history,
                    model=self.model_name,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                    stop=stop
                )
                return chat_completion.choices[0].message.content
            except (PermissionDeniedError, APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                print(f"带历史的对话调用失败（第{retry_count + 1}次重试）：{str(e)}")
                retry_count += 1
                if retry_count % self.max_retries_per_api == 0:
                    self._switch_next_api()
                time.sleep(1)
        print(f"所有API尝试失败，无法处理带历史的对话")
        return None