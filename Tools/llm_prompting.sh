#!/bin/bash




# models=("qwen2.5:72b" "b")
# models=("llama3.1:8b" "llama3.1:70b" "llama3.2:1b" "llama3.2:3b" "llama3.3:70b" "qwen2.5:72b" "mistral-nemo:12b" "mistral-small:24b" "mistral-large:123b") 
models=("llama3.1:8b" "llama3.1:70b" "llama3.2:1b" "llama3.2:3b" "llama3.3:70b" "qwen2.5:72b" "mistral-nemo:12b" "mistral-large:123b") 
# models=("mistral-small:24b")


# for model in "${models[@]}"; do
#     echo "Processing: ${model}"
#     python llm_prompting.py --api-type ollama --data large_scale_dataframes/multicw-test.csv --model "${model}" --prompt_type GA
#     python llm_prompting.py --api-type ollama --data large_scale_dataframes/multicw-test.csv --model "${model}" --prompt_type CoT_CLEF_on_Q
# done

# python llm_prompting.py --api-type openai --data large_scale_dataframes/multicw-test.csv --model o4-mini --prompt_type GA
# python llm_prompting.py --api-type openai --data large_scale_dataframes/multicw-test.csv --model o4-mini --prompt_type CoT_CLEF_on_Q



# OPENAI
# python llm_prompting.py --api-type openai --data large_scale_dataframes/multicw-test.csv --model gpt-4.1 --prompt_type GA
# python llm_prompting.py --api-type openai --data large_scale_dataframes/multicw-test.csv --model gpt-4.1 --prompt_type CoT_CLEF_on_Q


# python llm_prompting.py --api-type openai --data large_scale_dataframes/multicw-test.csv --model gpt-4o --prompt_type GA
# python llm_prompting.py --api-type openai --data large_scale_dataframes/multicw-test.csv --model gpt-4o --prompt_type CoT_CLEF_on_Q



# ANTROPIC
# python llm_prompting.py --api-type anthropic --data large_scale_dataframes/multicw-test.csv --model claude-3-5-haiku-20241022 --prompt_type GA
python llm_prompting.py --api-type anthropic --data large_scale_dataframes/multicw-test.csv --model claude-3-5-haiku-20241022 --prompt_type CoT_CLEF_on_Q

python llm_prompting.py --api-type anthropic --data large_scale_dataframes/multicw-test.csv --model claude-3-7-sonnet-20250219 --prompt_type CoT_CLEF_on_Q
python llm_prompting.py --api-type anthropic --data large_scale_dataframes/multicw-test.csv --model claude-3-7-sonnet-20250219 --prompt_type GA
