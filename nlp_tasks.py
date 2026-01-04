import streamlit as st
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="NLP Tasks Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .task-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 NLP Tasks Hub</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=200)
    st.title("Navigation")
    task = st.selectbox(
        "Choose an NLP Task:",
        [
            "🎭 Sentiment Analysis",
            "🏷️ Zero-Shot Classification",
            "✍️ Text Generation",
            "🎯 Mask Filling",
            "👤 Named Entity Recognition",
            "❓ Question Answering",
            "📝 Summarization",
            "🌐 Translation"
        ]
    )
    
    st.markdown("---")
    st.info("**About**: This app demonstrates 8 different NLP tasks using Hugging Face Transformers")
    st.markdown("**Powered by**: 🤗 Transformers")

# Initialize session state for caching pipelines
if 'pipelines' not in st.session_state:
    st.session_state.pipelines = {}

def get_pipeline(task_name, model=None):
    """Cache and retrieve pipelines"""
    cache_key = f"{task_name}_{model if model else 'default'}"
    if cache_key not in st.session_state.pipelines:
        with st.spinner(f'Loading {task_name} model...'):
            if model:
                st.session_state.pipelines[cache_key] = pipeline(task_name, model=model)
            else:
                st.session_state.pipelines[cache_key] = pipeline(task_name)
    return st.session_state.pipelines[cache_key]

# Task 1: Sentiment Analysis
if task == "🎭 Sentiment Analysis":
    st.header("🎭 Sentiment Analysis")
    st.write("Analyze the sentiment (positive/negative) of your text")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")
    
    if analyze_btn and text_input:
        classifier = get_pipeline('sentiment-analysis')
        result = classifier(text_input)
        
        st.markdown("### Results:")
        sentiment = result[0]['label']
        score = result[0]['score']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Confidence", f"{score:.2%}")
        
        # Progress bar for confidence
        st.progress(score)

# Task 2: Zero-Shot Classification
elif task == "🏷️ Zero-Shot Classification":
    st.header("🏷️ Zero-Shot Classification")
    st.write("Classify text into custom categories without training")
    
    text_input = st.text_area(
        "Enter text to classify:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    labels_input = st.text_input(
        "Enter candidate labels (comma-separated):",
        placeholder="e.g., education, entertainment, politics, gaming, art"
    )
    
    if st.button("🏷️ Classify", type="primary") and text_input and labels_input:
        labels = [label.strip() for label in labels_input.split(',')]
        classifier = get_pipeline('zero-shot-classification')
        result = classifier(text_input, candidate_labels=labels)
        
        st.markdown("### Classification Results:")
        for label, score in zip(result['labels'], result['scores']):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{label}**")
                st.progress(score)
            with col2:
                st.metric("Score", f"{score:.2%}")

# Task 3: Text Generation
elif task == "✍️ Text Generation":
    st.header("✍️ Text Generation")
    st.write("Generate text continuation using GPT-2")
    
    text_input = st.text_area(
        "Enter prompt:",
        height=100,
        placeholder="Start typing something..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max length:", 50, 200, 100)
    with col2:
        num_return = st.slider("Number of sequences:", 1, 3, 1)
    
    if st.button("✨ Generate", type="primary") and text_input:
        generator = get_pipeline('text-generation', model='gpt2')
        results = generator(text_input, max_length=max_length, num_return_sequences=num_return)
        
        st.markdown("### Generated Text:")
        for i, result in enumerate(results):
            with st.expander(f"Generation {i+1}", expanded=(i==0)):
                st.write(result['generated_text'])

# Task 4: Mask Filling
elif task == "🎯 Mask Filling":
    st.header("🎯 Mask Filling")
    st.write("Fill in the [MASK] token in your sentence")
    
    st.info("💡 Tip: Use [MASK] where you want the model to predict a word")
    
    text_input = st.text_input(
        "Enter text with [MASK]:",
        placeholder="e.g., I am learning [MASK] in this course."
    )
    
    if st.button("🎯 Fill Mask", type="primary") and text_input:
        if '[MASK]' not in text_input:
            st.error("Please include [MASK] in your text!")
        else:
            unmasker = get_pipeline('fill-mask', model='google-bert/bert-base-uncased')
            results = unmasker(text_input)
            
            st.markdown("### Top Predictions:")
            for i, result in enumerate(results[:5]):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"**#{i+1}**")
                with col2:
                    st.write(f"**{result['token_str']}**")
                    st.caption(result['sequence'])
                with col3:
                    st.metric("Score", f"{result['score']:.2%}")

# Task 5: Named Entity Recognition
elif task == "👤 Named Entity Recognition":
    st.header("👤 Named Entity Recognition (NER)")
    st.write("Extract named entities (persons, locations, organizations) from text")
    
    text_input = st.text_area(
        "Enter text:",
        height=150,
        placeholder="e.g., I am Alex. I am from Austria. It rains diamonds on Neptune."
    )
    
    if st.button("🔍 Extract Entities", type="primary") and text_input:
        ner = get_pipeline('ner')
        results = ner(text_input)
        
        st.markdown("### Detected Entities:")
        if results:
            for result in results:
                entity_type = result['entity'].replace('I-', '').replace('B-', '')
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                with col1:
                    st.write(f"**{result['word']}**")
                with col2:
                    st.write(f"`{entity_type}`")
                with col3:
                    st.caption(f"Position: {result['start']}-{result['end']}")
                with col4:
                    st.metric("", f"{result['score']:.2%}")
        else:
            st.info("No entities detected in the text.")

# Task 6: Question Answering
elif task == "❓ Question Answering":
    st.header("❓ Question Answering")
    st.write("Ask questions about a given context")
    
    context = st.text_area(
        "Context:",
        height=150,
        placeholder="Provide the context text here..."
    )
    
    question = st.text_input(
        "Question:",
        placeholder="What do you want to know about the context?"
    )
    
    if st.button("💡 Get Answer", type="primary") and context and question:
        qa = get_pipeline('question-answering')
        result = qa(question=question, context=context)
        
        st.markdown("### Answer:")
        st.success(f"**{result['answer']}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['score']:.2%}")
        with col2:
            st.caption(f"Position in context: {result['start']}-{result['end']}")

# Task 7: Summarization
elif task == "📝 Summarization":
    st.header("📝 Text Summarization")
    st.write("Generate concise summaries of longer texts")
    
    text_input = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Paste a long text here..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max summary length:", 20, 150, 50)
    with col2:
        min_length = st.slider("Min summary length:", 10, 50, 20)
    
    if st.button("📝 Summarize", type="primary") and text_input:
        summarizer = get_pipeline('summarization', model='facebook/bart-large-cnn')
        result = summarizer(text_input, max_length=max_length, min_length=min_length)
        
        st.markdown("### Summary:")
        st.info(result[0]['summary_text'])
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original words", len(text_input.split()))
        with col2:
            st.metric("Summary words", len(result[0]['summary_text'].split()))
        with col3:
            reduction = (1 - len(result[0]['summary_text'].split()) / len(text_input.split())) * 100
            st.metric("Reduction", f"{reduction:.1f}%")

# Task 8: Translation
elif task == "🌐 Translation":
    st.header("🌐 Translation")
    st.write("Translate text from English to French")
    
    text_input = st.text_area(
        "Enter English text:",
        height=150,
        placeholder="Type text to translate..."
    )
    
    if st.button("🌐 Translate", type="primary") and text_input:
        translator = get_pipeline('translation', model='Helsinki-NLP/opus-mt-en-fr')
        result = translator(text_input)
        
        st.markdown("### Translation:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**English:**")
            st.info(text_input)
        
        with col2:
            st.markdown("**French:**")
            st.success(result[0]['translation_text'])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with ❤️ using Streamlit and 🤗 Transformers</p>
        <p>All models are loaded from Hugging Face Model Hub</p>
    </div>
    """,
    unsafe_allow_html=True
)