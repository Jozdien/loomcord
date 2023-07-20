import discord
from discord.ext import commands
import openai
import tiktoken
from api_details import api_base, api_key
import json

with open("discord-token.txt", "r") as f:
    DISCORD_TOKEN = f.read().strip()
openai.api_key = api_key
openai.api_base = api_base

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
bot = commands.Bot(command_prefix="!", intents=intents)

CONTEXT_WINDOW = 8000
ARG_LIST = {
    "--loom-server": 0,
    "--exclude-names": False,
    "--num_children": 5,
    "--max_tokens": 50,
    "--temperature": 0.7,
    }
MODEL_ARGS = ["--num_children", "--max_tokens", "--temperature"]



@bot.event
async def on_message(message):
    if bot.user.mentioned_in(message) and message.author != bot.user:
        content = message.content.replace(f'<@{bot.user.id}>', '').strip()
        arg_values, content = check_arguments(content, ARG_LIST)
        model_args = get_model_args(arg_values)
        stop_sequences = None

        if arg_values["--loom-server"]:
            last_messages = list(reversed(await get_last_n_messages(message, arg_values["--loom-server"])))
            if not arg_values.get("--exclude-names"):
                content = '\n---\n'.join([f"{list(d.keys())[0]}: {list(d.values())[0]}" for d in last_messages]) + "\n---\n"
                stop_sequences = "\n---\n"
            else:
                content = '\n\n'.join([list(d.values())[0] for d in last_messages])
        else:
            content = await read_attachments(message, content)

        content, num_tokens = context_window(content,
                                             CONTEXT_WINDOW - model_args["--max_tokens"],
                                             encoding_name="gpt2")
        print(f"Number of tokens: {num_tokens}")

        continuations = get_gpt3_continuations(content, model_args, stop_sequences=stop_sequences)
        embeds, view = create_components(continuations)

        persist_args = persist_args_string(model_args)

        if len(content) + len(persist_args) > 2000:
            with open("response.txt", "w") as response_file:
                response_file.write(content)
            with open("response.txt", "rb") as response_file:
                await message.reply(content='' + persist_args, embeds=embeds, view=view,
                                    file=discord.File(response_file),
                                    mention_author=False)
        else:
            await message.reply(content + persist_args, embeds=embeds, view=view)

@bot.event
async def on_interaction(interaction):
    if isinstance(interaction, discord.Interaction) and interaction.data['component_type'] == 2:  # Corresponds to button click
        await interaction.response.defer()  # necessary to show interaction hasn't failed
        option_selected = int(interaction.data["custom_id"])
        partial_content, model_args = read_persist_args(interaction.message.content)
        original_content = await read_attachments(interaction.message, partial_content)
        content = f'{original_content}{interaction.message.embeds[option_selected].fields[0].value}'
        content, num_tokens = context_window(content,
                                             CONTEXT_WINDOW-model_args["--max_tokens"],
                                             encoding_name="gpt2")
        print(f"Number of tokens: {num_tokens}")

        continuations = get_gpt3_continuations(content, model_args)

        embeds, view = create_components(continuations)

        persist_args = persist_args_string(model_args)

        if len(content) + len(persist_args) > 2000:
            with open("response.txt", "w") as response_file:
                response_file.write(content)
            with open("response.txt", "rb") as response_file:
                await interaction.message.reply(content='' + persist_args, embeds=embeds, view=view, file=discord.File(response_file), mention_author=False)
        else:
            await interaction.message.reply(content + persist_args, embeds=embeds, view=view, mention_author=False)

def persist_args_string(model_args):
    args = json.dumps(model_args)
    return f"\n```===settings===\n{args}```"

def read_persist_args(content):
    separator = "\n```===settings==="
    if separator in content:
        content_end_idx = content.index(separator)
        start_idx = content.index(separator) + len(separator) + 1
        args = json.loads(content[start_idx:].replace("```", ""))
        return content[:content_end_idx], args
    return content, {a: ARG_LIST[a] for a in MODEL_ARGS}

def get_model_args(arg_values):
    return {a: arg_values[a] for a in MODEL_ARGS}

def get_gpt3_continuations(prompt, model_args, stop_sequences=None):
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompt,
        max_tokens=model_args["--max_tokens"],
        n=model_args["--num_children"],
        stop=stop_sequences,
        temperature=model_args["--temperature"],
    )

    continuations = [choice.text for choice in response.choices]
    # continuations = [str(i) for i in range(model_args["--num_children"])]
    return continuations

def check_arguments(input_string, arg_list):
    parts = input_string.split()
    arg_values = arg_list.copy()

    index = 0
    while index < len(parts):
        if parts[index] == "--loom-server":
            arg_values["--loom-server"] = int(parts[index + 1]) if index + 1 < len(parts) else 5
            index += 2
        elif parts[index] == "--exclude-names":
            arg_values["--exclude-names"] = True
            index += 1
        elif index + 1 < len(parts):
            if parts[index] in ["--num_children", "-n"]:
                arg_values["--num_children"] = int(parts[index+1])
            elif parts[index] in ["--max_tokens", "-m"]:
                arg_values["--max_tokens"] = int(parts[index+1])
            elif parts[index] in ["--temperature", "-t"]:
                arg_values["--temperature"] = float(parts[index+1])
            else:
                break
            index += 2
        else:
            break

    rest_of_string = " ".join(parts[index:])
    return arg_values, rest_of_string

def context_window(prompt, size, encoding_name="p50k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens, token_integers, token_bytes = get_tokens(prompt, encoding=encoding)
    if num_tokens > size:
        return encoding.decode(token_integers[-size:]), size
    return encoding.decode(token_integers), num_tokens

def get_tokens(prompt, encoding):
    token_integers = encoding.encode(prompt)
    num_tokens = len(token_integers)
    token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
    return num_tokens, token_integers, token_bytes

async def read_attachments(message, input_content, test=False):
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.endswith('.txt'):
                file_content = await attachment.read()
                if input_content:
                    input_content = f"{input_content}\n\n\{file_content.decode('utf-8')}"
                else:
                    input_content = file_content.decode('utf-8')
    if test:
        return True
    return input_content

async def get_last_n_messages(message, n):
    previous_messages = []
    async for msg in message.channel.history(before=message, limit=n):
        msg_dict = {msg.author.name: msg.content}
        previous_messages.append(msg_dict)
    return previous_messages

def create_components(continuations):
    embeds = []
    embed = discord.Embed(title="Choose continuation:")
    for i, c in enumerate(continuations):
        embed = discord.Embed()
        embed.add_field(name=f"Child {i+1}", value=f"\u200b{c}")
        embeds.append(embed)

    buttons = [
        [discord.ui.Button(label=f"Child {i+1}", custom_id=str(i))] for i, _  in enumerate(continuations)
    ]

    view = discord.ui.View()
    for button in buttons:
        view.add_item(*button)

    return embeds, view

bot.run(DISCORD_TOKEN)
