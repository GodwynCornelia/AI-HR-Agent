import json
import os
import time
from functools import lru_cache
from pathlib import Path
import httpx

from .limits import integer


@lru_cache(maxsize=2)
def tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def protect_role_boundaries(text):
    # Содержимое пользователя/документов не может вставить управляющий токен Qwen.
    return text.replace('<|','‹|').replace('|>','|›')


class Ollama:
    def __init__(self):
        self.deadline=None
        self.chat_calls=0
        self.chat_events=[]
        self.base = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
        self.model = os.getenv('LLM_MODEL', 'qwen2.5:1.5b')
        self.embed_model = os.getenv('EMBED_MODEL', 'embeddinggemma')
        self.keep_alive_seconds=max(0,int(os.getenv('MODEL_KEEP_ALIVE_SECONDS','0')))
        self.embed_keep_alive_seconds=max(0,int(os.getenv('EMBED_KEEP_ALIVE_SECONDS','0')))
        self.embed_options={'num_gpu':int(os.environ['EMBED_GPU_LAYERS'])} if 'EMBED_GPU_LAYERS' in os.environ else {}

    def fits(self, system, messages):
        path = os.getenv('TOKENIZER_PATH', str(Path(__file__).resolve().parent.parent / 'models/qwen-tokenizer.json'))
        encoder = tokenizer(path)
        contents = [system] + [message['content'] for message in messages]
        # Запас на служебные токены Qwen chat template и весь выход.
        tokens = sum(len(encoder.encode(content, add_special_tokens=False).ids) + 32 for content in contents) + 128
        return tokens + integer('MODEL_OUTPUT', 700) <= integer('MODEL_CONTEXT', 4096)

    def embed(self, texts, *, query=False, title='none'):
        if self.embed_model.startswith('embeddinggemma'):
            prefix = 'task: search result | query: ' if query else f'title: {title} | text: '
            texts = [prefix + text for text in texts]
        with httpx.Client(timeout=self.request_timeout(300)) as client:
            payload={'model':self.embed_model,'input':texts,'truncate':False,'keep_alive':self.embed_keep_alive_seconds,'options':self.embed_options}
            self.last_embed_request=payload
            response = client.post(self.base + '/api/embed', json=payload)
            response.raise_for_status()
            data = response.json()
            vectors = data['embeddings']
            self.last_embed_metadata = {key: value for key, value in data.items() if key != 'embeddings'}
        if len(vectors) != len(texts):
            raise ValueError('Число embeddings не совпало с числом текстов')
        return vectors

    def chat(self, system, messages, structured=True):
        self.last_raw_response=''
        self.last_chat_metadata={}
        messages=[{**message,'content':protect_role_boundaries(message['content'])} for message in messages]
        if not self.fits(system, messages):
            raise ValueError('Вопрос и контекст превышают окно модели; сократите вопрос или историю.')
        data = {'model': self.model, 'messages': [{'role': 'system', 'content': system}] + messages,
                'stream': False, 'keep_alive': self.keep_alive_seconds,
                'options': {'temperature': 0, 'seed': 116, 'num_ctx': integer('MODEL_CONTEXT', 4096),
                            'num_predict': integer('MODEL_OUTPUT', 700), 'num_gpu':max(0,int(os.getenv('MODEL_GPU_LAYERS','0')))}}
        if self.model.startswith('qwen3'):
            data['think']=False
        if structured:
            data['format'] = structured if isinstance(structured,dict) else 'json'
        self.last_request = data
        with httpx.Client(timeout=self.request_timeout(600)) as client:
            self.chat_calls+=1
            response = client.post(self.base + '/api/chat', json=data)
            response.raise_for_status()
        data = response.json()
        self.last_chat_metadata = {key: value for key, value in data.items() if key != 'message'}
        self.chat_events.append(self.last_chat_metadata.copy())
        content = data['message']['content']
        self.last_raw_response=content
        return json.loads(content) if structured else content

    def request_timeout(self,default):
        if self.deadline is None:return default
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Общий бюджет времени исчерпан')
        return min(default,remaining)
