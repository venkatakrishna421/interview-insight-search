import os
import json
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Insight Search",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
  .main { background-color: #f8f9fb; }
  .title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
  .subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
  .result-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #4F8EF7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .speaker { font-weight: 600; color: #4F8EF7; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .timestamp { color: #999; font-size: 0.8rem; margin-left: 0.5rem; }
  .quote { color: #1a1a2e; font-size: 0.95rem; line-height: 1.6; margin-top: 0.4rem; }
  .score { color: #aaa; font-size: 0.75rem; margin-top: 0.5rem; }
  .insight-box {
    background: #eef4ff;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.5rem;
    border: 1px solid #c7d9f8;
    font-size: 0.95rem;
    color: #2c3e6b;
    line-height: 1.7;
  }
  .tag {
    display: inline-block;
    background: #e8f0fe;
    color: #3b5bdb;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
    margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ── Sample transcripts (realistic customer research interviews) ─
SAMPLE_TRANSCRIPTS = [
    {
        "interview": "Interview 1 – Sarah (Product Manager, Fintech)",
        "turns": [
            {"speaker": "Interviewer", "time": "0:32", "text": "What's your biggest pain point when running customer research today?"},
            {"speaker": "Sarah", "time": "0:45", "text": "Honestly, the biggest frustration is that insights just disappear after the interview. We record everything but nobody goes back and watches 45 minute videos. So decisions get made without the actual customer voice."},
            {"speaker": "Interviewer", "time": "1:10", "text": "How do you currently share findings with your team?"},
            {"speaker": "Sarah", "time": "1:18", "text": "Usually a slide deck. But by the time I've made the deck, I've already filtered and summarized so much that the raw customer emotion is completely lost. My stakeholders never hear the actual words customers used."},
            {"speaker": "Interviewer", "time": "2:05", "text": "Have you tried any tools to help with this?"},
            {"speaker": "Sarah", "time": "2:12", "text": "We tried Dovetail for a while but the pricing was really hard to justify to our finance team. It was like 400 dollars a month and leadership just didn't see the ROI. We ended up canceling after two months."},
            {"speaker": "Interviewer", "time": "3:00", "text": "What would the ideal solution look like for you?"},
            {"speaker": "Sarah", "time": "3:09", "text": "Something where I can just search across all my past interviews like Google. I want to type 'pricing objections' and instantly see every moment a customer talked about price. That would save me hours every week."},
        ]
    },
    {
        "interview": "Interview 2 – Marcus (UX Researcher, E-commerce)",
        "turns": [
            {"speaker": "Interviewer", "time": "0:15", "text": "Tell me about your current research workflow."},
            {"speaker": "Marcus", "time": "0:22", "text": "We do about 8 to 10 user interviews per sprint. The problem is I'm the only researcher so I end up being a bottleneck. Engineers and PMs want insights fast but transcription and analysis takes me a full day per interview."},
            {"speaker": "Interviewer", "time": "1:30", "text": "Where do insights tend to get lost?"},
            {"speaker": "Marcus", "time": "1:38", "text": "In the gap between the interview and the report. I take notes but they're always incomplete. I miss things. And when I go back to the recording two weeks later, I can never find the exact moment where the user said that one really important thing."},
            {"speaker": "Interviewer", "time": "2:20", "text": "How do your stakeholders react to research findings?"},
            {"speaker": "Marcus", "time": "2:28", "text": "They love quotes. Actual verbatim quotes from users are way more convincing than any chart I can make. But surfacing those quotes is incredibly manual right now. I'm literally scrubbing through video."},
            {"speaker": "Interviewer", "time": "3:15", "text": "What about recruiting participants?"},
            {"speaker": "Marcus", "time": "3:20", "text": "That's another nightmare. We use a spreadsheet to track who we've talked to, when, what segment they're in. It's completely unscalable. I've heard Great Question has good recruiting tools but haven't tried it yet."},
            {"speaker": "Interviewer", "time": "4:00", "text": "What would make your job 10x easier?"},
            {"speaker": "Marcus", "time": "4:07", "text": "Automatic theme detection across interviews. If I could say show me every time someone mentioned checkout friction across 50 interviews, that would be transformative. I'd actually have time to do analysis instead of just logistics."},
        ]
    },
    {
        "interview": "Interview 3 – Priya (Head of Design, SaaS)",
        "turns": [
            {"speaker": "Interviewer", "time": "0:20", "text": "How does research feed into your design process?"},
            {"speaker": "Priya", "time": "0:28", "text": "In theory, every design decision should be backed by research. In practice, we're moving so fast that we often design based on assumptions and then do research to validate after. Which kind of defeats the purpose."},
            {"speaker": "Interviewer", "time": "1:15", "text": "What's the biggest gap in your current research stack?"},
            {"speaker": "Priya", "time": "1:22", "text": "Connecting qualitative insights to product decisions. I can run a great interview and get amazing insights but then those insights sit in a Notion doc that nobody reads. There's no connection to the actual tickets in Jira or the designs in Figma."},
            {"speaker": "Interviewer", "time": "2:10", "text": "How do you handle consent and participant data?"},
            {"speaker": "Priya", "time": "2:18", "text": "It's really manual and honestly a bit scary from a compliance standpoint. We have a Google Form for consent, recordings in Drive, notes in Notion, and participant info in a spreadsheet. If we ever got audited I'd be very nervous."},
            {"speaker": "Interviewer", "time": "3:00", "text": "What features would you pay more for?"},
            {"speaker": "Priya", "time": "3:08", "text": "An AI that could watch all my past interviews and tell me what themes are emerging. And the ability to share a highlight reel with stakeholders without me having to manually clip video. Those two things would make research feel less like a bottleneck."},
            {"speaker": "Interviewer", "time": "3:55", "text": "Any concerns about AI being involved in research?"},
            {"speaker": "Priya", "time": "4:02", "text": "Bias. I worry that AI summarization loses the nuance and emotion in customer feedback. A customer might say something was fine but their tone clearly meant it wasn't. I'd want AI to assist not replace human judgment."},
        ]
    },
    {
        "interview": "Interview 4 – Jordan (Startup Founder)",
        "turns": [
            {"speaker": "Interviewer", "time": "0:10", "text": "How often are you doing customer interviews right now?"},
            {"speaker": "Jordan", "time": "0:16", "text": "Maybe 2 or 3 a week when things are going well. But honestly I deprioritize it when we get busy which is exactly backwards. Customer research is most important when you're moving fast and making big decisions."},
            {"speaker": "Interviewer", "time": "1:00", "text": "What's stopping you from doing more?"},
            {"speaker": "Jordan", "time": "1:06", "text": "Time and follow-through. I'll do the interview and feel energized but then I don't have a system for extracting and applying the insights. They just live in my head and fade. I need something that forces the workflow."},
            {"speaker": "Interviewer", "time": "2:00", "text": "Have you tried any research tools?"},
            {"speaker": "Jordan", "time": "2:05", "text": "I've tried Maze for usability testing and it was great for that specific use case. But for generative research, open ended interviews, I haven't found anything that fits early stage startup workflows. Most tools feel built for enterprise research teams."},
            {"speaker": "Interviewer", "time": "2:55", "text": "What would make you pay for a research tool?"},
            {"speaker": "Jordan", "time": "3:02", "text": "If it could turn a 45 minute interview into a 2 minute brief that I could share with my co-founder and investors. Right now I'm the only one who gets value from interviews because I'm the only one who listens to them. That's a huge problem."},
            {"speaker": "Interviewer", "time": "3:50", "text": "Any thoughts on AI moderated interviews?"},
            {"speaker": "Jordan", "time": "3:58", "text": "I'm actually really excited about this. If an AI could run a solid discovery interview at 2am when a user is willing to give feedback, that's incredible scale. The question is whether the conversation feels natural enough that users open up."},
        ]
    },
]

# ── Helpers ────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

@st.cache_resource(show_spinner=False)
def build_index():
    """Embed all transcript turns and build FAISS index."""
    chunks = []
    for interview in SAMPLE_TRANSCRIPTS:
        for turn in interview["turns"]:
            chunks.append({
                "interview": interview["interview"],
                "speaker":   turn["speaker"],
                "time":      turn["time"],
                "text":      turn["text"],
            })

    with st.spinner("🔧 Building search index from interview transcripts..."):
        embeddings = [get_embedding(c["text"]) for c in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vecs
    matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index.add(matrix)
    return index, chunks

def search(query: str, index, chunks: list, top_k: int = 5):
    q_emb = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({**chunks[idx], "score": float(score)})
    return results

def generate_insight(query: str, results: list) -> str:
    context = "\n".join(
        f"[{r['interview']} | {r['speaker']} @ {r['time']}]: {r['text']}"
        for r in results
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are a senior UX researcher analyzing customer interview transcripts. "
                "Given search results for a researcher's query, write a concise 2-3 sentence "
                "synthesis of the key insight. Be specific, quote exact language where useful, "
                "and highlight any patterns across multiple participants. Be direct and actionable."
            )},
            {"role": "user", "content": f"Query: {query}\n\nRelevant transcript excerpts:\n{context}"}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content

# ── UI ─────────────────────────────────────────────────────────

st.markdown('<div class="title">🔍 Interview Insight Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Semantic search across customer interview transcripts — '
    'find what your customers actually said, instantly.</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Loaded Interviews")
    for t in SAMPLE_TRANSCRIPTS:
        st.markdown(f"- {t['interview']}")
    st.markdown("---")
    st.markdown("### 💡 Try searching for:")
    example_queries = [
        "pricing concerns",
        "insights getting lost",
        "AI moderated interviews",
        "sharing research with stakeholders",
        "tool is too expensive",
        "manual and time consuming",
        "search across past interviews",
    ]
    for q in example_queries:
        if st.button(q, key=q):
            st.session_state["query"] = q
    st.markdown("---")
    top_k = st.slider("Results to show", min_value=2, max_value=8, value=4)

# Build index once
index, chunks = build_index()

# Search bar
query = st.text_input(
    "Search across all interviews",
    value=st.session_state.get("query", ""),
    placeholder='e.g. "pricing objections" or "too much manual work"',
    key="search_input"
)

if query:
    with st.spinner("Searching..."):
        results = search(query, index, chunks, top_k=top_k)

    if results:
        # AI synthesis
        with st.spinner("Generating insight summary..."):
            insight = generate_insight(query, results)

        st.markdown("#### ✨ AI Insight")
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown(f"#### 📌 Top {len(results)} matching moments")
        for r in results:
            similarity_pct = int(r["score"] * 100)
            st.markdown(f"""
            <div class="result-card">
              <span class="speaker">{r['speaker']}</span>
              <span class="timestamp">@ {r['time']} · {r['interview']}</span>
              <div class="quote">"{r['text']}"</div>
              <div class="score">Relevance: {similarity_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No results found. Try a different search term.")
else:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What this does")
        st.markdown("""
        - Embeds all interview transcript turns using **OpenAI text-embedding-3-small**
        - Stores vectors in a **FAISS index** for fast similarity search
        - Returns the most semantically relevant moments — even if the exact words don't match
        - Generates an **AI insight summary** across the top results using GPT-4o
        """)
    with col2:
        st.markdown("#### Why it matters")
        st.markdown("""
        - Researchers waste hours scrubbing through recordings to find key moments
        - Keyword search misses paraphrased or contextually similar quotes
        - This surfaces the **actual customer voice** in seconds
        - Scales to thousands of hours of interviews
        """)