import streamlit as st
import json
import yaml
from pathlib import Path
from datasets import load_dataset

# Set wide layout
st.set_page_config(layout="wide")

LOCAL_FILE = "local_changes.json"

# Load metadata only
@st.cache_data
def load_dataset_metadata():
    with st.spinner("Loading dataset..."):
        dataset = load_dataset("LuckyLukke/negotio_REFUEL_test", split="train")
    return dataset


def parse_conversation(conversation_str):
    """Parse conversation string - try JSON first, otherwise return as-is."""
    if not conversation_str:
        return None
    
    # Try to parse as JSON (list of messages)
    try:
        parsed = json.loads(conversation_str)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If not JSON, return as string
    return conversation_str


@st.cache_data
def load_game_config(game_type, game_settings):
    """Load game config YAML file and return parties."""
    # Only game_settings is required, game_type is optional
    if not game_settings:
        return None
    
    # Construct path to game config file
    # game_settings might already have .yaml extension or not
    game_settings_name = game_settings
    if not game_settings_name.endswith('.yaml'):
        game_settings_name = f"{game_settings_name}.yaml"
    
    # Try to find the config file
    # Path structure: envs/negotiation/configs/games/{game_settings}.yaml
    # Use absolute path from script location
    script_dir = Path(__file__).parent.parent  # Go up from display/ to root
    config_path = script_dir / "envs" / "negotiation" / "configs" / "games" / game_settings_name
    
    if not config_path.exists():
        # Try relative path as fallback
        config_path_rel = Path("envs/negotiation/configs/games") / game_settings_name
        if config_path_rel.exists():
            config_path = config_path_rel
        else:
            st.warning(f"Config file not found at {config_path} or {config_path_rel}")
            return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('parties', [])
    except Exception as e:
        st.warning(f"Could not load game config: {e}")
        return None


def get_conversation_data(dataset, index):
    row = dataset[index]
    return {
        "sampled_h": row.get("sampled_h", None),
        "starting_agent": row.get("starting_agent", None),
        "negotiation_role": row.get("negotiation_role", None),
        "game_settings": row.get("game_settings", None),
        "issues": row.get("issues", []),
        "issue_weights": row.get("issue_weights", []),
        "chosen": row.get("chosen", None),
        "reject": row.get("reject", None),
        "chosen_reward": row.get("chosen_reward", None),
        "reject_reward": row.get("reject_reward", None),
        "chosen_generated_tokens": row.get("chosen_generated_tokens", None),
        "reject_generated_tokens": row.get("reject_generated_tokens", None),
    }

# Load dataset
dataset = load_dataset_metadata()
num_conversations = len(dataset)

# Initialize session state for current index
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Top navigation for conversation index
st.title(f"Conversation {st.session_state['current_index'] + 1} / {num_conversations}")
col_prev, col_current, col_next = st.columns([1, 4, 1])

with col_prev:
    if st.button("Previous") and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
with col_next:
    if st.button("Next") and st.session_state.current_index < num_conversations - 1:
        st.session_state.current_index += 1

with col_current:
    selected_index = st.selectbox(
        "Jump to Conversation:",
        options=list(range(num_conversations)),
        format_func=lambda x: f"Conversation {x + 1}",
        index=st.session_state.current_index,
    )
    st.session_state.current_index = selected_index

# Fetch selected conversation
data = get_conversation_data(dataset, st.session_state["current_index"])

# Display metadata at the top
if data:
    # Parse conversations
    chosen_conv = parse_conversation(data.get("chosen"))
    reject_conv = parse_conversation(data.get("reject"))
    
    # Display metadata
    st.write("### Metadata")
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.write(f"**Sampled H:** {data.get('sampled_h', 'N/A')}")
        st.write(f"**Game:** {data.get('game_settings', 'N/A')}")
    with col_meta2:
        st.write(f"**Starting Agent:** {data.get('starting_agent', 'N/A')}")
        st.write(f"**Role:** {data.get('negotiation_role', 'N/A')}")
    with col_meta3:
        if data.get('issues'):
            st.write(f"**Issues:**")
            for issue in data['issues']:
                st.write(f"- {issue}")
        else:
            st.write(f"**Issues:** N/A")
    with col_meta4:
        if data.get('issue_weights'):
            st.write(f"**Issue Weights:**")
            issue_weights = data['issue_weights']
            issues = data.get('issues', [])
            # Display weights for each agent
            if isinstance(issue_weights, list) and len(issue_weights) > 0:
                if isinstance(issue_weights[0], list):
                    # Format: [[agent1_weights], [agent2_weights]]
                    for agent_idx, weights in enumerate(issue_weights):
                        agent_name = f"Agent {agent_idx + 1}"
                        weight_str = ", ".join([f"{w}" for w in weights])
                        st.write(f"**{agent_name}:** {weight_str}")
                else:
                    # Single list of weights
                    weight_str = ", ".join([f"{w}" for w in issue_weights])
                    st.write(weight_str)
        else:
            st.write(f"**Issue Weights:** N/A")

    # Load game config to get parties dynamically
    game_type = data.get('game_type')
    game_settings = data.get('game_settings')
    parties = load_game_config(game_type, game_settings)
    
    # Determine roles based on negotiation_role
    # If negotiation_role is 1, assistant is first party, user is second party
    # Otherwise, assistant is second party, user is first party
    negotiation_role = data.get('negotiation_role')
    
    if parties and len(parties) >= 2:
        # Handle both string and int negotiation_role
        if negotiation_role == 1 or negotiation_role == "1" or str(negotiation_role) == "1":
            assistant_party = parties[0]  # First party (e.g., Landlord)
            user_party = parties[1]  # Second party (e.g., Tenant)
        else:
            assistant_party = parties[1]  # Second party (e.g., Tenant)
            user_party = parties[0]  # First party (e.g., Landlord)
    else:
        # Fallback to default if parties not found
        assistant_party = "Assistant"
        user_party = "User"
    
    roles = {"assistant": assistant_party, "user": user_party, "system": "System"}
    # Icons: assistant is always bot, user is always human
    avatars = {"assistant": "🤖", "user": "👤", "system": "🤖"}

    def get_background_color(index, is_highlighted):
        if is_highlighted:
            return "#5e1e25"  # Light red for highlighting
        return "#161617" if index % 2 == 0 else "#414142"  # Alternating grey shaades

    def display_messages(messages, sampled_h=None, is_chosen=True):
        """Display messages from a list or string."""
        if messages is None:
            st.write("No conversation data available.")
            return
        
        # If messages is a list, display as structured messages
        if isinstance(messages, list):
            # Find the assistant message at position sampled_h + 1 (where divergence occurs)
            # sampled_h counts assistant messages, so sampled_h=2 means highlight the 3rd assistant message
            divergence_index = None
            if not is_chosen and sampled_h is not None:
                assistant_count = 0
                for i, msg in enumerate(messages):
                    if msg.get("role") == "assistant":
                        assistant_count += 1
                        # If we've passed sampled_h assistant messages, this is the one to highlight
                        if assistant_count == sampled_h + 1:
                            divergence_index = i
                            break
            
            for i, msg in enumerate(messages):
                # Highlight the first assistant message after sampled_h + 1 for reject
                is_highlighted = not is_chosen and i == divergence_index
                background_color = get_background_color(i, is_highlighted)
                style = f"background-color: {background_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;"
                
                if msg.get("role") == "system":
                    pass
                elif msg.get("role") == "user":
                    with st.container():
                        st.markdown(
                            f"<div style='{style}'><strong>{avatars['user']} {roles['user']}:</strong> {msg.get('content', '').replace('$', '&#36;')}</div>",
                            unsafe_allow_html=True,
                        )
                elif msg.get("role") == "assistant":
                    with st.container():
                        st.markdown(
                            f"<div style='{style}'><strong>{avatars['assistant']} {roles['assistant']}:</strong> {msg.get('content', '').replace('$', '&#36;')}</div>",
                            unsafe_allow_html=True,
                        )
        else:
            # If it's a string, display as text
            st.text_area("Conversation", messages, height=400, key=f"conv_{'chosen' if is_chosen else 'reject'}")

    # Display both conversations side by side
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chosen (Better)")
        
        # Display reward and token info
        if data.get("chosen_reward") is not None:
            st.write(f"**Reward:** {data['chosen_reward']:.4f}")
        if data.get("chosen_generated_tokens") is not None:
            st.write(f"**Generated Tokens:** {data['chosen_generated_tokens']}")
        
        display_messages(chosen_conv, sampled_h=None, is_chosen=True)

    with col2:
        st.subheader("Reject (Worse)")
        
        # Display reward and token info
        if data.get("reject_reward") is not None:
            st.write(f"**Reward:** {data['reject_reward']:.4f}")
        if data.get("reject_generated_tokens") is not None:
            st.write(f"**Generated Tokens:** {data['reject_generated_tokens']}")

        
        display_messages(reject_conv, sampled_h=data.get("sampled_h"), is_chosen=False)
else:
    st.error("No conversation found.")
