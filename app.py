import streamlit as st
from classifier import graph, State
from memory import save_to_memory, get_similar_problems
st.set_page_config(page_title="Math Mentor", page_icon="📐", layout="wide")
st.title("📚🤯IIT JEE Math Mentor")
st.caption("Powered by RAG + Multi-Agent AI")

# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = None
if "query" not in st.session_state:
    st.session_state.query = ""

# ------------------------------------------------------------------
# Input Mode
# ------------------------------------------------------------------
st.caption("Text input is currently enabled while image and audio processing are being restored.")
query = st.text_area("Enter your math question", height=100)

# ------------------------------------------------------------------
# Solve Button
# ------------------------------------------------------------------
if st.button("🧠 Solve", disabled=not query.strip()):
    st.session_state.feedback_given = None  # reset feedback on new solve
    with st.spinner("Running agents..."):
        state: State = {
            "query": query,
            "route_label": None,
            "final_answer": None,
            "retrieved_context": None,
            "verification_notes": None,
            "explanation": None,
            "needs_hitl": None,
            "verified": None,
            "parsed_problem": None,
            "total_cost": 0.0
        }
        st.session_state.result = graph.invoke(state)
        st.session_state.query = query

# ------------------------------------------------------------------
# Show results if we have them
# ------------------------------------------------------------------
if st.session_state.result:
    result = st.session_state.result

    # HITL
    if result["needs_hitl"]:
        st.warning("⚠️ Verifier is not confident. Please review the answer below.")
        corrected = st.text_area(
            "Edit the answer if needed, then click Confirm",
            value=result["final_answer"],
            height=200,
            key="hitl_edit"
        )
        if st.button("✅ Confirm & Continue"):
            st.session_state.result["final_answer"] = corrected
            st.success("Answer confirmed!")

    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Solution")
        st.markdown(result["final_answer"])
        verified_color = "🟢" if result["verified"] else "🔴"
        st.caption(f"{verified_color} Verifier: {result['verification_notes']}")
    with col2:
        st.subheader("🎓 Explanation")
        st.markdown(result["explanation"])

    # Agent Trace
    with st.expander("🔍 Agent Trace"):
        st.markdown(f"**💰 Cost this request:** `${result['total_cost']:.6f}`")
        st.markdown("**Parsed Problem:**")
        st.json(result["parsed_problem"])
        st.markdown(f"**Topic classified:** `{result['route_label']}`")
        st.markdown(f"**Verified:** `{result['verified']}`")
        st.markdown(f"**Verifier notes:** {result['verification_notes']}")
        st.markdown(f"**HITL triggered:** `{result['needs_hitl']}`")

    # Retrieved Context
    with st.expander("📚 Retrieved Formulas from Knowledge Base"):
        st.markdown(result["retrieved_context"])

    # Similar Problems
    similar = get_similar_problems(st.session_state.query)
    if similar:
        with st.expander("🕐 Similar Problems Solved Before"):
            for i, prob in enumerate(similar, 1):
                st.markdown(f"**{i}. {prob['original_query']}**")
                st.markdown(f"Answer: {prob['final_answer'][:200]}...")
                st.divider()

    # Feedback
    st.divider()
    st.markdown("**Was this answer helpful?**")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("✅ Correct", key="correct_btn"):
            if save_to_memory(result, feedback="correct"):
                st.session_state.feedback_given = "correct"
            else:
                st.warning("Memory is temporarily unavailable, so this solution was not saved.")
    with col4:
        if st.button("❌ Incorrect", key="incorrect_btn"):
            st.session_state.feedback_given = "incorrect"

    if st.session_state.feedback_given == "correct":
        st.success("Saved to memory! Thanks for the feedback.")
    elif st.session_state.feedback_given == "incorrect":
        comment = st.text_input("What was wrong?", key="incorrect_comment")
        if st.button("Submit feedback", key="submit_feedback"):
            if comment:
                if save_to_memory(result, feedback=f"incorrect: {comment}"):
                    st.info("Feedback recorded. We'll improve!")
                else:
                    st.warning("Memory is temporarily unavailable, so this feedback was not saved.")
