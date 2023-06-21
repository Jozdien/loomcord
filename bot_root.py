import discord
from discord.ext import commands
import openai
import tiktoken
from api_details import api_base, api_key

with open("discord-token.txt", "r") as f:
    DISCORD_TOKEN = f.read().strip()
openai.api_key = api_key
openai.api_base = api_base

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
bot = commands.Bot(command_prefix="!", intents=intents)

CONTEXT_WINDOW = 8000
ARG_LIST = {"--loom-server": 0, "--exclude-names": False}

@bot.event
async def on_message(message):
    if bot.user.mentioned_in(message) and message.author != bot.user:
        content = message.content.replace(f'<@{bot.user.id}>', '').strip()
        arg_values, content = check_arguments(content, ARG_LIST)
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

        content, num_tokens = context_window(content, CONTEXT_WINDOW-50, encoding_name="gpt2")
        print(f"Number of tokens: {num_tokens}")

        continuations = get_gpt3_continuations(content, stop_sequences=stop_sequences)
        embeds, view = create_components(continuations)

        if len(content) > 2000:
            with open("response.txt", "w") as response_file:
                response_file.write(content)
            with open("response.txt", "rb") as response_file:
                await message.reply(content='', embeds=embeds, view=view, file=discord.File(response_file), mention_author=False)
        else:
            await message.reply(content, embeds=embeds, view=view)

@bot.event
async def on_interaction(interaction):
    if isinstance(interaction, discord.Interaction) and interaction.data['component_type'] == 2:  # Corresponds to button click
        await interaction.response.defer()  # necessary to show interaction hasn't failed
        option_selected = int(interaction.data["custom_id"])
        original_content = await read_attachments(interaction.message, interaction.message.content)
        content = f'{original_content}{interaction.message.embeds[option_selected].fields[0].value}'
        content, num_tokens = context_window(content, CONTEXT_WINDOW-50, encoding_name="gpt2")
        print(f"Number of tokens: {num_tokens}")

        continuations = get_gpt3_continuations(content)

        embeds, view = create_components(continuations)

        if len(content) > 2000:
            with open("response.txt", "w") as response_file:
                response_file.write(content)
            with open("response.txt", "rb") as response_file:
                await interaction.message.reply(content='', embeds=embeds, view=view, file=discord.File(response_file), mention_author=False)
        else:
            await interaction.message.reply(content, embeds=embeds, view=view, mention_author=False)

def get_gpt3_continuations(prompt, stop_sequences=None):
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=5,
        stop=stop_sequences,
        temperature=0.7,
    )

    continuations = [choice.text for choice in response.choices]
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
            continue
        if index >= len(parts) or parts[index] not in arg_list:
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