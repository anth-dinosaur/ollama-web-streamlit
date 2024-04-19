import os

from dotenv import load_dotenv
from ollama import Client
import streamlit as st
from streamlit_pills import pills
import requests

load_dotenv()


def update_stats():
    with stats_box.container():
        st.header("Generation statistics")
        if "gen_stats" in st.session_state:
            st.write(
                f"Total generated tokens: {st.session_state.gen_stats.get('total_tokens',0)}"
            )
            st.write(
                f"Avg speed: {round(st.session_state.gen_stats.get('total_tokens',0)/st.session_state.gen_stats.get('total_seconds',1),1)} tokens/s"
            )
        else:
            st.write("No generations performed yet")
    update_glances_metrics()


def get_glances_api_data(endpoint):
    try:
        response = requests.get(
            f"http://{st.session_state.server}:61208/api/3/{endpoint}"
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get data from {endpoint}: {e}")
        return None


def update_glances_metrics():
    cpu_data = get_glances_api_data("cpu/total")
    cpu_percent = round(cpu_data.get("total", 0), 1) if cpu_data else 0

    mem_data = get_glances_api_data("mem/percent")
    mem_percent = round(mem_data.get("percent", 0), 0) if mem_data else 0

    gpu_data = get_glances_api_data("gpu/gpu_id/0")
    if gpu_data and "0" in gpu_data:
        gpu_info = gpu_data["0"][0]
        gpu_percent = round(gpu_info.get("proc", 0), 1)
        gpu_temp = round(gpu_info.get("temperature", 0), 0)
        gpu_mem_percent = round(gpu_info.get("mem", 0), 0)
    else:
        gpu_percent = gpu_temp = gpu_mem_percent = 0

    with system_box.container():
        st.header("System stats")
        pills(
            label="Glances data",
            options=[
                f"CPU: {cpu_percent}%",
                f"RAM: {mem_percent}%",
                f"GPU: {gpu_percent}%",
                f"GPU Mem: {gpu_mem_percent}%",
                f"GPU Temp: {gpu_temp}°C",
            ],
            index=None,
            key="system_stats_pills",
        )


st.title("Ollama Web Interface")

with st.sidebar:
    st.text_input(
        "IP address of Ollama server:",
        key="server",
        value=os.getenv("OLLAMA_SERVER", ""),
    )
    stats_box = st.empty()
    system_box = st.empty()
    update_stats()

if "server" in st.session_state and st.session_state.server.strip() != "":
    try:
        # Initialize Ollama client
        ollama = Client(f"http://{st.session_state.server}:11434")

        # Create a dropdown for model selection
        models = [model["name"] for model in ollama.list()["models"]]
        default_model = os.getenv("OLLAMA_MODEL", "")
        selected_model = pills(
            label="Select a model:",
            options=models,
            index=models.index(default_model) if default_model in models else 0,
            key="pills",
        )
    except Exception as e:
        st.error(f"Failed to initialize Ollama: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Say something..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Talking to a robot..."):
            # Generate a chat reposnse
            try:
                response_placeholder = st.empty()

                full_response = ""
                for chunk in ollama.chat(
                    model=selected_model,
                    messages=st.session_state.messages,
                    stream=True,
                ):
                    full_response += chunk["message"]["content"]
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

                if "gen_stats" not in st.session_state:
                    st.session_state["gen_stats"] = {
                        "total_tokens": 0,
                        "total_seconds": 0,
                    }
                st.session_state.gen_stats["total_tokens"] += chunk["eval_count"]
                st.session_state.gen_stats["total_seconds"] += (
                    chunk["eval_duration"] / 1000000000
                )
                update_stats()
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
