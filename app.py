import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, blankline_tokenize, WhitespaceTokenizer, wordpunct_tokenize
from nltk.util import bigrams, trigrams, ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure punkt is downloaded
nltk.download('punkt')

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main app styling */
    .stButton button {
        background-color: rgb(41, 183, 83);
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: rgb(31, 141, 49);
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
        color: black !important; /* Change text color to black */
    }
    .stApp {
        max-width: 1200px;
        margin: auto;
    }

    /* Sidebar styling with dark background */
    .css-1d391kg {
        background-color: rgb(40, 40, 40); /* Dark background */
        padding: 20px;
        border-radius: 10px;
        color: white !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: rgb(255, 255, 255) !important; /* White text for headings */
    }
    .css-1d391kg .stSelectbox, .css-1d391kg .stFileUploader, .css-1d391kg .stSlider {
        background-color: rgba(79, 78, 96, 0.1);
        border-radius: 8px;
        padding: 10px;
        color: white !important;
    }
    .css-1d391kg .stSelectbox label, .css-1d391kg .stFileUploader label, .css-1d391kg .stSlider label {
        color: white !important;
    }
    .css-1d391kg .stSelectbox div[role="button"] {
        color: white !important;
    }
    .css-1d391kg .st-bb {
        border-color: white !important;
    }

    /* Dark background for "MY Interests" section */
    .sidebar .sidebar-content {
        background-color: rgb(40, 40, 40); /* Dark background */
        color: rgb(255, 255, 255); /* White text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px;">
        <h1 style="text-align: center; color: #333;">NLP Text Analysis Application</h1>
        <p style="text-align: center; color: #666;">Analyze your text with ease!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for settings with dark background
st.sidebar.header("MY Interests")

# Dark background and white text for sidebar
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: rgb(40, 40, 40); /* Dark background */
        color: rgb(255, 255, 255); /* White text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Theme toggle
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: rgb(30, 30, 30);
            color: rgb(255, 255, 255);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])
user_input = ""
if uploaded_file is not None:
    user_input = uploaded_file.read().decode("utf-8")
    user_input = st.text_area("Edit your text:", value=user_input)
else:
    user_input = st.text_area("Enter your text:")

# Perform tokenization only if input is provided
if user_input.strip():
    word_tokens = word_tokenize(user_input)
    sent_tokens = sent_tokenize(user_input)
    blankline_tokens = blankline_tokenize(user_input)
    whitespace_tokens = WhitespaceTokenizer().tokenize(user_input)
    punctuation_tokens = wordpunct_tokenize(user_input)

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Tokenization", "N-grams", "Visualization"])

    with tab1:
        st.subheader("Tokenization")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Word Tokens"):
                st.write("Word Tokens:", word_tokens)
            if st.button("Sentence Tokens"):
                st.write("Sentence Tokens:", sent_tokens)
        with col2:
            if st.button("Blankline Tokens"):
                st.write("Blankline Tokens:", blankline_tokens)
            if st.button("Whitespace Tokens"):
                st.write("Whitespace Tokens:", whitespace_tokens)
            if st.button("Punctuation Tokens"):
                st.write("Punctuation Tokens:", punctuation_tokens)

    with tab2:
        st.subheader("N-grams")
        if st.button("Bigrams"):
            st.write("Bigrams:", list(bigrams(word_tokens)))
        if st.button("Trigrams"):
            st.write("Trigrams:", list(trigrams(word_tokens)))
        ngrams_input = st.number_input("Enter value of n for Ngrams", min_value=1, step=1, value=2)
        if st.button("Generate Ngrams"):
            ngram_tokens = list(ngrams(word_tokens, int(ngrams_input)))
            st.write(f"{ngrams_input}-grams:", ngram_tokens)

    with tab3:
        st.subheader("WordCloud")
        max_words = st.slider("Max Words in WordCloud", min_value=10, max_value=200, value=100)
        if st.button("Generate WordCloud"):
            plt.clf()
            wc = WordCloud(max_words=max_words, margin=1, background_color='black', colormap='Accent', mode='RGBA').generate(user_input)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

else:
    st.warning("Please enter some text to analyze.")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; color: pink;">
        <p>Developed by Purnima | Powered by Streamlit and NLTK</p>
    </div>
    """,
    unsafe_allow_html=True
)