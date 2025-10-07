import argparse
import os

import pandas as pd
import anthropic
import ollama
from pandas import DataFrame
from tqdm import tqdm
from ollama import Client
from enum import Enum
from sklearn.metrics import classification_report
import time
from functools import partial
from typing import Callable
from openai import OpenAI

# Load .env variables
from dotenv import load_dotenv
load_dotenv() 

# Enable tqdm for pandas
tqdm.pandas()

class ApiType(Enum):
    openai = 'openai'
    anthropic = 'anthropic'
    ollama = 'ollama'
    lm_studio = 'lm_studio'
    nim_nvidia = 'nim_nvidia'


NIM_NVIDIA_API_KEY = 'TBA'
# ANTHROPIC_API_KEY = 'TBA'

def create_conversation_GA(text: str) -> str:
    return [
        {
             "role": "system",
             "content": """You are a fact-checker assistant with a task to identify
 sentences that are check-worthy. Sentence is check-worthy only if it contains a verifiable 
 factual claim and that claim can be harmful."""
        },
        {
             "role": "user",
             "content": f"""
Classify the check-worthiness of these sentences outputting only Yes or No:
[{text}]
do not display any explanations."""
        }
    ]
    
def create_conversation_CoT_CLEF_on_Q(text: str) -> str:
    return [
        {
             "role": "system",
             "content": """You will be presented with a text and asked to determine if it contains a check-worthy claim. To make this determination, you will follow a series of steps and answer a set of questions.
Your final answer will be "Yes" if the text contains a check-worthy claim and "No" if it does not."""
        },
        {
             "role": "user",
             "content": f"""Text: [{text}]

Chain of Thoughts (CoT) Steps:

Step 1: Read the text carefully and identify any claims or statements that could be verified or debunked by a fact-checker.

Step 2: Consider the likelihood that the claim in the text is false or misleading. Ask yourself: Is the claim suspicious or too good (or bad) to be true?

Step 3: Evaluate the significance of the claim. Is it a matter of public interest or a trivial/personal matter? Would the general public be interested in knowing whether the claim is true or false?

Step 4: Assess the potential impact of the claim if it is false or misleading. Could it harm individuals, organizations, or society as a whole?

Step 5: Determine if the text appears to be spreading rumours or misinformation about a particular topic or individual.

Step 6: Check if the claim is supported by credible sources or evidence. Are there any references or citations provided to back up the claim?

Step 7: Analyse the tone and language used in the text. Is it emotive or sensationalist? Could it be intended to manipulate or deceive readers?

Step 8: Consider the topic of the claim. Is it related to a matter of public concern, such as healthcare, politics, or current events?

Step 9: Evaluate the potential harm that the claim could cause if it is false or misleading. Could it be used to discredit individuals, organizations, or products?

Step 10: Finally, assess whether the text provides enough context and information for a reader to make an informed decision about the claim's validity.

Final Step:

Based on your answers to the previous steps, determine if the text contains a check-worthy claim. If you answered "yes" to any of the following questions - 1, 4, 5, 7, 8, or 9 - or if you answered "no" to questions 6 or 10, then the text likely contains a check-worthy claim. Otherwise, the text does not contain a check-worthy claim.

Response Format:

Please respond with a simple "Yes" or "No" to indicate whether the text contains a check-worthy claim. Do not display any explanations.
"""
        }
    ]


def followup(model_response: str) -> list[dict[str, str]]:
    conversation = []
    conversation.append({"role": "system", "content": "Does it mean Yes or No? Answer strictly with Yes or No. If are unsure, answer with No."})
    conversation.append({"role": "user", "content": model_response})
    return conversation
            

def infer(model_api, text: str) -> str:
    wait = 1
    while True:
        try:
            time.sleep(0.05)
            if args.prompt_type == "GA":
                conversation = create_conversation_GA(text)
            elif args.prompt_type == "CoT_CLEF_on_Q":
                conversation = create_conversation_CoT_CLEF_on_Q(text)
                
            response = model_api(conversation).strip().lower()
            if 'yes' in response:
                return True
            elif 'no' in response:
                return False
            second_response = model_api(followup(response)).strip().lower()
            return 'yes' in second_response
        except Exception as e:
            print(f"{e}")
            time.sleep(wait)
            wait *= 2


def init_model() -> Callable[[list[dict[str, str]]], str]:
    if args.api_type == ApiType.openai:
        client = OpenAI()
        return lambda conversation: client.chat.completions.create(
            model=args.model,
            messages=conversation,
            # temperature=0.1,
        ).choices[0].message.content

    if args.api_type == ApiType.anthropic:
        client = anthropic.Anthropic()
        def antropic_fn(conversation):
            for c in conversation:
                if c["role"] == 'system':
                    c["role"] = 'assistant'
            return client.messages.create(
                model=args.model,
                messages=conversation,
                max_tokens=5000,
            ).content[0].text
        return antropic_fn

    if args.api_type == ApiType.ollama:
        return lambda conversation: ollama.chat(
            model=args.model,
            messages=conversation,
        )['message']['content']

    if args.api_type == ApiType.lm_studio:
        pass
        # TODO:
        # completion = OpenAI(base_url="http://localhost:1234/v1",
        #                     api_key="lm-studio").chat.completions.create(
        #     model=model,
        #     messages=message,
        #     temperature=0.1,
        # )

    if args.api_type == ApiType.nim_nvidia:
        pass
        # TODO:
        # completion = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
        #                     api_key=NIM_NVIDIA_API_KEY).chat.completions.create(
        #     model=model,
        #     messages=message,
        #     temperature=0.1,
        # )


def main():
    df = pd.read_csv(args.data)
    key = f'fc_worthy_{args.prompt_type}_{args.model}'
    if key in df.columns:
        print(f"Column `{key}` already exists")
        return

    infer_fn = partial(infer, init_model())
    df[key] = df['text'].progress_apply(infer_fn)
    df.to_csv(args.data, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='large_scale_dataframes/multicw-test-small-test.csv')
    parser.add_argument('--api-type', type=ApiType, choices=list(ApiType), required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prompt_type', type=str, required=True)
    # parser.add_argument('--output', type=str, default='Final-dataset/multicw-test_with-predictions.csv')
    args = parser.parse_args()

    assert args.prompt_type in ("GA", "CoT_CLEF_on_Q")
    main()

