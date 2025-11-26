import os
import streamlit as st
import rlcard
import torch
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
import pandas as pd

# Try rule-based bot
try:
    from rlcard.agents import LimitholdemRuleAgent
    RULE_AGENT_AVAILABLE = True
except Exception:
    RULE_AGENT_AVAILABLE = False


# -------------------------------------------------------------
# ENV + BOT
# -------------------------------------------------------------
def create_env():
    """Create the Limit Texas Hold'em environment."""
    env = rlcard.make(
        "limit-holdem",
        config={
            "seed": 42,
            "allow_raw_data": True,  # we want raw_obs to display cards nicely
        },
    )
    return env


def create_bot(env):
    """Create the bot. Prefer loading trained DQN agent saved with torch."""
    model_path = "models/dqn_limit_holdem.pth"

    # Try to load trained DQN agent
    if os.path.exists(model_path):
        try:
            agent = torch.load(model_path, map_location="cpu")
            print("✅ Loaded trained DQNAgent from", model_path)
            return agent
        except Exception as e:
            print("⚠ Could not load trained agent, reason:", e)

    # Fallbacks (no training)
    if RULE_AGENT_AVAILABLE:
        print("➡ Using rule-based LimitHoldem agent instead.")
        return LimitholdemRuleAgent(num_actions=env.num_actions)

    print("➡ Using random agent (no trained/rule agent available).")
    return RandomAgent(num_actions=env.num_actions)


# -------------------------------------------------------------
# GAME MANAGEMENT
# -------------------------------------------------------------
def reset_game():
    env = st.session_state.env
    state, player = env.reset()

    st.session_state.state = state
    st.session_state.current_player = player
    st.session_state.done = False
    st.session_state.payoffs = None
    st.session_state.step = 0
    st.session_state.history = []  # will store both human and bot moves


# -------------------------------------------------------------
# RECORD HISTORY (dataset-style table)
# -------------------------------------------------------------
def record_state(last_action=None, actor=None):
    """
    Save one row of game history.

    last_action: string like 'call', 'fold', etc.
    actor: 'Human', 'Bot', or None (for initial state).
    """
    raw = st.session_state.state.get("raw_obs", {})
    st.session_state.history.append(
        {
            "step": st.session_state.step,
            "player_turn": st.session_state.current_player,
            "actor": actor,
            "action": last_action,
            "hand": ", ".join(raw.get("hand", [])),
            "public_cards": ", ".join(raw.get("public_cards", [])),
            "pot": raw.get("pot"),
            "chips": raw.get("my_chips"),
        }
    )


# -------------------------------------------------------------
# BOT AUTO PLAY
# -------------------------------------------------------------
def auto_bot_play():
    env = st.session_state.env
    bot = st.session_state.bot
    bot_pos = st.session_state.bot_position

    # Let bot act up to 20 times in a row (if it's its turn)
    for _ in range(20):
        if st.session_state.done:
            break
        if st.session_state.current_player != bot_pos:
            break

        action = bot.step(st.session_state.state)
        action_label = env._decode_action(action)

        next_state, next_player = env.step(action)

        st.session_state.state = next_state
        st.session_state.current_player = next_player
        st.session_state.step += 1
        record_state(last_action=action_label, actor="Bot")

        if env.is_over():
            st.session_state.done = True
            st.session_state.payoffs = env.get_payoffs()
            break


# -------------------------------------------------------------
# CARD VISUAL DISPLAY
# -------------------------------------------------------------
def draw_cards(cards):
    if not cards:
        return "No cards"

    suit_map = {
        "S": "♠",
        "H": "♥",
        "D": "♦",
        "C": "♣",
    }

    visual = []

    for c in cards:
        c = c.strip()

        # Case 1: Normal (Rank + Suit) → e.g. "AS", "9D"
        if len(c) == 2 and c[1] in suit_map:
            rank = c[0]
            suit = suit_map[c[1]]
            visual.append(f"{rank}{suit}")
            continue

        # Case 2: Suit + Rank → "HA", "D9"
        if len(c) == 2 and c[0] in suit_map:
            suit = suit_map[c[0]]
            rank = c[1]
            visual.append(f"{rank}{suit}")
            continue

        # Case 3: Hidden
        if "?" in c:
            visual.append("🂠")
            continue

        # Fallback
        visual.append(f"{c}")

    return " | ".join(visual)


# -------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title="Poker Bot", layout="wide")
    st.title("🃏 Poker Bot – Human vs Trained RL Agent")

    # Initialize once
    if "init" not in st.session_state:
        set_seed(42)
        st.session_state.env = create_env()

        st.session_state.human_position = 0  # you = player 0
        st.session_state.bot_position = 1    # bot = player 1

        st.session_state.bot = create_bot(st.session_state.env)

        st.session_state.init = True

        reset_game()
        # initial state history (no actor/action yet)
        record_state(last_action=None, actor=None)
        auto_bot_play()

    # 3 tabs (includes graph tab)
    tab1, tab2, tab3 = st.tabs(["🎮 Play Game", "📊 Game History", "📈 Training & Comparison"])

    # ---------------------------------------------------------
    # TAB 1 — GAMEPLAY
    # ---------------------------------------------------------
    with tab1:
        st.header("🎮 Play Poker vs Bot")

        raw = st.session_state.state.get("raw_obs", {})
        hand = raw.get("hand", [])
        public = raw.get("public_cards", [])
        pot = raw.get("pot")
        chips = raw.get("my_chips")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Cards")
            st.markdown(draw_cards(hand))

        with col2:
            st.subheader("Board Cards")
            st.markdown(draw_cards(public))

        st.write(f"*Pot:* {pot}")
        st.write(f"*Your Chips:* {chips}")

        # ------------------------
        # GAME OVER
        # ------------------------
        if st.session_state.done:
            pay = st.session_state.payoffs
            st.success("✔ Game Over")

            st.write(f"Your Payoff: *{pay[0]}*")
            st.write(f"Bot Payoff: *{pay[1]}*")

            if pay[0] > pay[1]:
                st.balloons()
                st.success("🎉 You Won!")
            elif pay[0] < pay[1]:
                st.error("🤖 Bot Wins!")
            else:
                st.warning("Draw.")

            if st.button("🔄 New Game"):
                reset_game()
                record_state(last_action=None, actor=None)
                auto_bot_play()
                st.rerun()

        # ------------------------
        # HUMAN TURN
        # ------------------------
        else:
            if st.session_state.current_player == st.session_state.human_position:
                st.info("👉 Your Turn")

                legal = list(st.session_state.state["legal_actions"])
                cols = st.columns(len(legal))
                picked = None
                picked_label = None

                for i, aid in enumerate(legal):
                    label = st.session_state.env._decode_action(aid)
                    if cols[i].button(label.upper()):
                        picked = aid
                        picked_label = label

                if picked is not None:
                    env = st.session_state.env
                    ns, np = env.step(picked)

                    st.session_state.state = ns
                    st.session_state.current_player = np
                    st.session_state.step += 1
                    record_state(last_action=picked_label, actor="Human")

                    if env.is_over():
                        st.session_state.done = True
                        st.session_state.payoffs = env.get_payoffs()
                    else:
                        auto_bot_play()

                    st.rerun()

            else:
                st.info("🤖 Bot is thinking...")
                auto_bot_play()
                st.rerun()

    # ---------------------------------------------------------
    # TAB 2 — GAME HISTORY TABLE
    # ---------------------------------------------------------
    with tab2:
        st.header("📊 Game State History (Human + Bot Moves)")

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, height=500, use_container_width=True)
        else:
            st.write("No history yet. Play a game first.")

    # ---------------------------------------------------------
    # TAB 3 — TRAINING + MODEL COMPARISON (GRAPH)
    # ---------------------------------------------------------
    with tab3:
        st.header("📈 RL Training & Model Comparison")

        # Training history
        history_path = "models/training_history.csv"
        if os.path.exists(history_path):
            st.subheader("Training History (DQN Agent)")
            train_df = pd.read_csv(history_path)
            st.dataframe(train_df, height=250, use_container_width=True)
            avg_payoff = train_df["payoff"].mean()
            st.write(f"Average payoff over {len(train_df)} training episodes: **{avg_payoff:.3f}**")
        else:
            st.write("Training history file not found. Run `python train_agent.py` first.")

        st.markdown("---")

        # Comparison graph: DQN vs baseline
        comp_path = "models/comparison_history.csv"
        if os.path.exists(comp_path):
            st.subheader("Model Comparison: DQN vs Baseline (Rule/Random)")

            comp_df = pd.read_csv(comp_path)
            pivot_df = comp_df.pivot(index="episode", columns="model", values="avg_payoff")

            st.line_chart(pivot_df)

            st.write("This graph shows the running **average payoff** per episode for:")
            st.write("- **DQN** (trained RL agent)")
            st.write("- **Baseline** model (rule-based if available, otherwise random agent)")
        else:
            st.write("Comparison history file not found. It is generated by `python train_agent.py`.")


if __name__ == "__main__":
    main()
