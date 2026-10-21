# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the expectations of the `chat` test group: conversations rendered by the chat templates of
Tests/Data/chat/*.jinja as Hugging Face's transformers renders them (apply_chat_template: jinja2's sandbox
with trim_blocks and lstrip_blocks, loop controls, its tojson, raise_exception and strftime_now), into
Tests/Data/chat_cases.txt:

    python Tests/make_chat_tests.py        # needs transformers and jinja2

Every case renders a template with a conversation: its messages, tools, add_generation_prompt, the special
tokens and flags a template reads (enable_thinking), and strftime_now() at a fixed time (TIME below, noon
UTC, so that every time zone within eleven hours has its date). The file holds, as lines:

    T name                      the template of Tests/Data/chat/name.jinja for the cases that follow
    C add time hex              a case: add_generation_prompt, the time, and the hexadecimal UTF-8 of a JSON
                                object of the variables (the special tokens, tools, flags, and the messages when
                                they are more than roles and texts)
    M role content              (hexadecimal) the messages of a case whose messages are roles and texts, which
                                the C test passes as SpingalettChatMessage
    E hex                       the text it renders to, or
    X hex                       the error's message (the template raised it, or refused the input)

The templates are those of the GGUF files and Hugging Face repositories of Qwen2.5, Qwen3, Llama 3.2,
Gemma 2 and 3, SmolLM2 and 3, Phi-3.5 and Phi-4 mini, Mistral 7B v0.3, DeepSeek-R1's distillations,
Granite 3.3, OLMo 2, GLM-4 and Falcon 3, as their authors publish them.
"""

import datetime
import glob
import json
import os

import transformers.utils.chat_template_utils as ctu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'Data')
TIME = 1767268800          # 2026-01-01 12:00 UTC


class FixedTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.datetime.fromtimestamp(TIME, tz)


ctu.datetime = FixedTime

# the special tokens of each template's tokenizer
TOKENS = {
    'qwen2.5': ('', '<|im_end|>'), 'qwen3': ('', '<|im_end|>'), 'llama3.2': ('<|begin_of_text|>', '<|eot_id|>'),
    'gemma3': ('<bos>', '<eos>'), 'gemma2': ('<bos>', '<eos>'), 'smollm2': ('<|im_start|>', '<|im_end|>'),
    'smollm3': ('', '<|im_end|>'), 'phi3.5': ('<s>', '<|endoftext|>'), 'phi4-mini': ('<|endoftext|>', '<|endoftext|>'),
    'mistral0.3': ('<s>', '</s>'), 'deepseek-r1-qwen': ('<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>'),
    'granite3.3': ('<|end_of_text|>', '<|end_of_text|>'), 'olmo2': ('<|endoftext|>', '<|endoftext|>'),
    'glm4-chat': ('', '<|endoftext|>'), 'glm4-0414': ('', '<|endoftext|>'), 'falcon3': ('<|endoftext|>', '<|endoftext|>'),
}

WEATHER = {'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'The weather in a city, now',
    'parameters': {'type': 'object', 'properties': {
        'city': {'type': 'string', 'description': 'The city, "Paris" or "Москва"'},
        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city']}}}


def conversations():
    """(simple, add_generation_prompt, variables) of every case."""
    yield True, True, {'messages': [{'role': 'user', 'content': 'Hello!'}]}
    yield True, True, {'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                                    {'role': 'user', 'content': 'What is 2+2?'}]}
    yield True, True, {'messages': [{'role': 'system', 'content': 'Answer briefly.'},
                                    {'role': 'user', 'content': 'What is 2+2?'},
                                    {'role': 'assistant', 'content': '4'},
                                    {'role': 'user', 'content': 'And 3+3?'}]}
    yield True, False, {'messages': [{'role': 'user', 'content': 'Hi'},
                                     {'role': 'assistant', 'content': 'Hello! How can I help?'}]}
    yield True, True, {'messages': [{'role': 'user', 'content': '  Привет, мир! 你好 🌍\n\tline two  '}]}
    yield True, True, {'messages': [{'role': 'user', 'content': 'quotes " \' \\ and {{ braces }} {% raw %}'}]}
    yield True, False, {'messages': [{'role': 'user', 'content': 'Think first.'},
                                     {'role': 'assistant', 'content': '<think>\nTwo and two.\n</think>\n\nFour.'}]}
    yield True, True, {'messages': [{'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}]}
    yield False, True, {'messages': [{'role': 'user', 'content': 'Hello!'}], 'enable_thinking': False}
    yield False, True, {'messages': [{'role': 'user', 'content': 'What is the weather in Paris?'}], 'tools': [WEATHER]}
    yield False, True, {'messages': [
        {'role': 'user', 'content': 'What is the weather in Paris?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {
            'name': 'get_weather', 'arguments': {'city': 'Paris', 'unit': 'celsius'}}}]},
        {'role': 'tool', 'name': 'get_weather', 'content': '{"temperature": 21.5, "sky": "clear"}'},
        {'role': 'assistant', 'content': 'It is 21.5 °C and clear.'},
        {'role': 'user', 'content': 'Thanks!'}], 'tools': [WEATHER]}
    yield False, True, {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Parts of a message'}]}]}


def main():
    lines = ['# Written by Tests/make_chat_tests.py: T template, C add time variables, M role content, E text or X error']
    count = 0
    for path in sorted(glob.glob(os.path.join(DATA, 'chat', '*.jinja'))):
        name = os.path.basename(path)[:-6]
        source = open(path, encoding='utf-8').read()
        template = ctu._cached_compile_jinja_template(source)
        bos, eos = TOKENS[name]
        lines.append('T ' + name)
        for simple, add, variables in conversations():
            kwargs = {'tools': None, 'documents': None, 'bos_token': bos, 'eos_token': eos}
            kwargs.update(variables)
            given = {'bos_token': bos, 'eos_token': eos}
            given.update({k: v for k, v in variables.items() if not (simple and k == 'messages')})
            text = json.dumps(given, ensure_ascii=False)
            lines.append('C %d %d %s' % (add, TIME, text.encode('utf-8').hex()))
            if simple:
                for m in variables['messages']:
                    lines.append('M %s %s' % (m['role'].encode('utf-8').hex(), m['content'].encode('utf-8').hex()))
            try:
                rendered = template.render(add_generation_prompt=add, **kwargs)
                lines.append('E ' + rendered.encode('utf-8').hex())
            except Exception as e:      # the template's raise_exception(), or a type it cannot take
                lines.append('X ' + str(e).encode('utf-8').hex())
            count += 1
    with open(os.path.join(DATA, 'chat_cases.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('chat_cases.txt: %d cases' % count)


if __name__ == '__main__':
    main()
