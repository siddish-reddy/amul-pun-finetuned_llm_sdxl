import os
import json

import asyncio
import dotenv
import aiohttp
import streamlit as st
import replicate

dotenv.load_dotenv()

OPENAI  = os.getenv('OPENAI')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# st.title("Amul GPT")
st.set_page_config(page_title="Amul GPT", layout="wide")

pun_few_shot = json.load(open('prompts/pun_few_shot.json'))
story_few_shot = json.load(open('prompts/story_few_shot.json'))
pun_finetune_static = json.load(open('prompts/pun_finetune_static.json'))

async def gpt_wrapper(messages, model):   
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI}",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "max_tokens": 1000,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as resp:
            print(resp.status)
            result = await resp.json()
            print(result)
            return result['choices'][0]['message']['content']


def main():
    # collect freeform text input headline
    headline_situation = st.text_input('Headline!', 'What happened?')
    submit_button = st.button('Submit')

    # after you collect the input, you can use it
    # to populate the widgets that the async function

    # layout your app beforehand, with st.empty
    # for the widgets that the async function would populate
    
    # turbo_3_5_finetune_output = st.empty()
    # turbo_4_128k_output = st.empty()

    # Create two columns
    col_headings = st.columns(2)
    col_headings[0].markdown('**Turbo 3.5 Finetune**')
    col_headings[1].markdown('**Turbo 4 128k**')
    col1, col2 = st.columns(2)

    # Place the widgets in the columns
    turbo_3_5_finetune_output = col1.empty()
    turbo_4_128k_output = col2.empty()

    st.write('---')
    st.markdown('**Image Caption**')
    
    image_caption_output = st.empty()

    st.write('---')
    st.markdown('**Image**')
    image = st.empty()

    if submit_button:
        try:
            # async run the draw function, sending in all the
            # widgets it needs to use/populate
            asyncio.run(draw_async(
                headline_situation,
                turbo_3_5_finetune_output,
                turbo_4_128k_output,
                image_caption_output,
                image
            ))
        except Exception as e:
            print(f'error...{type(e)}')
            raise
        finally:    
            # some additional code to handle user clicking stop
            print('finally')


async def get_turbo_3_5_finetune_output(turbo_3_5_finetune_output, headline_situation):
    model = "ft:gpt-3.5-turbo-1106:wh-ai::8JfzB6Z2"
    current_turn = {
        "role": "user",
        "content": f"Situation: {headline_situation}"
    }
    messages = pun_finetune_static + [current_turn]
    generation = await gpt_wrapper(messages, model)
    generation = generation.replace('Mainline:', '##')
    generation = generation.replace('Tagline:', '###')
    turbo_3_5_finetune_output.markdown(generation)

    return generation

async def get_turbo_4_128k_output(turbo_4_128k_output, headline_situation):
    model = "gpt-4-1106-preview"
    current_turn = {
        "role": "user",
        "content": f"Situation: {headline_situation}"
    }
    messages = pun_few_shot + [current_turn]
    generation = await gpt_wrapper(messages, model)
    generation = generation.replace('Mainline:', '##')
    generation = generation.replace('Tagline:', '###')
    turbo_4_128k_output.markdown(generation)
    return generation

async def get_image_caption_output(headline_situation):
    # do some work
    user_turn = {
        "role": "user",
        "content": f"Situation: {headline_situation}"
    }
    messages = story_few_shot + [user_turn]
    model = "gpt-4-1106-preview"
    generation = await gpt_wrapper(messages, model)
    return generation.strip()

async def get_image(image_caption_output, image, headline_situation):
    # do some work
    caption = await get_image_caption_output(headline_situation)
    image_caption_output.markdown(caption + "\n\n---\nGenerating image... (30s-90s)")

    output = await replicate.async_run(
        "zylim0702/sdxl-lora-customize-model:5a2b1cff79a2cf60d2a498b424795a90e26b7a3992fbd13b340f73ff4942b81e",
        input={
            "width": 400,
            "height": 400,
            "prompt": "a photo of " + caption.replace("Amul girl", "TOK").replace("amul girl", "TOK").replace("girl", "TOK"),
            "refine": "expert_ensemble_refiner",
            "Lora_url": "https://replicate.delivery/pbxt/RLwauYvGEqIPI9FVirNw66BZ2RWgy0xOVFYbm4cFCpmjozdE/trained_model.tar",
            "scheduler": "K_EULER",
            "lora_scale": 1,
            "num_outputs": 1,
            "guidance_scale": 7.5,
            # "apply_watermark": True,
            "high_noise_frac": 1,
            "negative_prompt": "text, caption",
            "prompt_strength": 0.8,
            "num_inference_steps": 250
        }
    )

    image.image(output[0])

    return output[0]


async def draw_async(
    headline_situation,
    turbo_3_5_finetune_output,
    turbo_4_128k_output,
    image_caption_output,
    image
):
    for i in range(3):
        task1 = get_turbo_3_5_finetune_output(turbo_3_5_finetune_output, headline_situation)
        task2 = get_turbo_4_128k_output(turbo_4_128k_output, headline_situation)
        task3 = get_image(image_caption_output, image, headline_situation)

        await asyncio.gather(task1, task2, task3)


if __name__ == '__main__':
    main()