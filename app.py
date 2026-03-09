import streamlit as st
from classifier import graph, State
import easyocr
import numpy as np
from PIL import Image
from memory import save_to_memory, get_similar_problems
import os
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
mode = st.radio("Input mode", ["Text", "Image", "Audio"], horizontal=True)

query = ""

if mode == "Text":
    query = st.text_area("Enter your math question", height=100)

elif mode == "Image":
    uploaded = st.file_uploader("Upload image of math problem", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)
        with st.spinner("Extracting text from image..."):
            reader = easyocr.Reader(['en'], gpu=False)
            ocr_result = reader.readtext(np.array(image), detail=0)
            extracted = " ".join(ocr_result)
        st.markdown("**Extracted text** (edit if needed):")
        query = st.text_area("Extracted text", value=extracted, height=100)
elif mode == "Audio":
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
    if audio_file:
        st.audio(audio_file)
        with st.spinner("Transcribing audio..."):
            import whisper
            import tempfile
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            # Transcribe
            model = whisper.load_model("tiny")
            result_whisper = model.transcribe(tmp_path)
            transcript = result_whisper["text"].strip()
            # Cleanup
            os.remove(tmp_path)
        st.markdown("**Transcript** (edit if needed):")
        query = st.text_area("Transcript", value=transcript, height=100)
        if result_whisper["segments"]:
            avg_confidence = sum(
                s.get("no_speech_prob", 0) for s in result_whisper["segments"]
            ) / len(result_whisper["segments"])
            if avg_confidence > 0.5:
                st.warning("⚠️ Low confidence transcript — please review carefully")        

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
            "parsed_problem": None
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
            save_to_memory(result, feedback="correct")
            st.session_state.feedback_given = "correct"
    with col4:
        if st.button("❌ Incorrect", key="incorrect_btn"):
            st.session_state.feedback_given = "incorrect"

    if st.session_state.feedback_given == "correct":
        st.success("Saved to memory! Thanks for the feedback.")
    elif st.session_state.feedback_given == "incorrect":
        comment = st.text_input("What was wrong?", key="incorrect_comment")
        if st.button("Submit feedback", key="submit_feedback"):
            if comment:
                save_to_memory(result, feedback=f"incorrect: {comment}")
                st.info("Feedback recorded. We'll improve!")