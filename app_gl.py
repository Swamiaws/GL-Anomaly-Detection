import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationSummaryBufferMemory

# Function to initialize session state
def init_state():
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.session_id = "unique_session_id"
    st.session_state.token_count = 0

# Function to select and initialize the LLM model
def select_llm_model(model_name, temperature):
    model_mapping = {
        "Gemma-7b-IT": "gemma-7b-it",
        "Llama3–70b-8192": "llama3-70b-8192",
        "Llama3–8b-8192": "llama3-8b-8192",
        "Mixtral-8x7b-32768": "mixtral-8x7b-32768"
    }
    selected_model = model_mapping.get(model_name)
    groq_api = st.secrets['GROQ_API_KEY']
    if not groq_api:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    llm = ChatGroq(temperature=temperature, model=selected_model, api_key=groq_api)
    return llm

# Function to convert Excel file to DataFrame
def convert_excel_to_df(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to create a pandas dataframe agent
def create_pandas_agent(llm, df, memory):
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        memory=memory,
        agent_type="tool-calling",
        verbose=True,
        prefix='You are interacting with a dataset named "Consolidated GL Sheet.xlsx" for GL anomaly detection. Provide detailed answers based on the data. If necessary, create concise code examples for visualization or analysis.',
        suffix='Only generate Python code if explicitly requested or if a visualization is required. Use `import streamlit as st` for Streamlit-based visualizations, and end with `st.pyplot(plt.gcf())` to display the plot.',
        allow_dangerous_code=True,
        include_df_in_prompt=True
    )
    return agent_executor

# Function to query the agent and extract output
def query_data(agent, query):
    response = agent.invoke(query)
    output_value = response.get('output', 'No output found')
    graph_code = response.get('graph_code', '').strip()
    st.session_state.token_count += response.get('token_usage', 0)
    return output_value, graph_code

# Set up the Streamlit page
st.set_page_config(page_title="GL Anomaly Chatbot", page_icon="📉", layout="wide")
st.title("GL Anomaly Chatbot")

# Initialize session state if not already present
if 'messages' not in st.session_state:
    init_state()

# Directly specify the file path
file_path = "Consolidated GL Sheet.xlsx" 
if file_path:
    with st.spinner("Loading data..."):
        try:
            # Convert file to DataFrame
            df = convert_excel_to_df(file_path)
            data_file_name = os.path.basename(file_path)

            # Sidebar for model selection
            with st.sidebar:
                st.subheader("Select LLM Model")
                selected_model = st.radio(
                    "Choose a model:",
                    ("Gemma-7b-IT", "Llama3–70b-8192", "Llama3–8b-8192", "Mixtral-8x7b-32768"),
                    index=1  # Default to Llama3-70b-8192
                )
                
                # Add temperature slider
                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

                # Display token usage and file name
                st.markdown(f"**Data File:** {data_file_name}")
                st.markdown(f"**Tokens Consumed:** {st.session_state.token_count}")

            if selected_model:
                # Initialize LLM
                llm = select_llm_model(selected_model, temperature)
                
                # Initialize memory with the LLM instance
                memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
                
                # Create the pandas agent with memory
                agent = create_pandas_agent(llm, df, memory)

                # Display existing messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

                # Handle user input and agent response
                if prompt := st.chat_input("Type your message here..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        placeholder.markdown("...")

                        try:
                            output_value, graph_code = query_data(agent, prompt)
                            placeholder.markdown(output_value, unsafe_allow_html=True)
                            st.session_state.messages.append({"role": "assistant", "content": output_value})

                            # Execute the graph code if it exists
                            if graph_code:
                                exec(graph_code)
                                st.pyplot(plt.gcf())
                            
                        except KeyError as e:
                            st.error(f"KeyError: {str(e)}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

# Display chat history if any
if st.session_state.history:
    for chat in st.session_state.history:
        st.write(f"**You:** {chat['query']}")
        st.write(f"**Agent:** {chat['response']}")
