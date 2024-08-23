import streamlit as st
import openai
import re
import os
from dotenv import load_dotenv
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Dict, List, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 下载必要的 NLTK 数据
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(page_title="Advanced Vocabulary Enhancement System", layout="wide")

# 设置 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 自定义 CSS
st.markdown("""
<style>
    .replaced-word {
        color: blue;
        text-decoration: underline;
        cursor: pointer;
    }
    .difficult-word {
        color: red;
        text-decoration: underline;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

difficulty_levels = ["C2", "C1", "B2", "B1", "A2", "A1"]

@st.cache_data
def chat(prompt: str, tmpr: float = 0.7, max_tokens: int = 300) -> Optional[str]:
    """
    向 ChatGPT 提交 prompt 并返回回应
    
    :param prompt: 提示文本
    :param tmpr: 温度参数
    :param max_tokens: 最大令牌数
    :return: ChatGPT 的回应或 None（如果出错）
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant specializing in vocabulary enhancement and context analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=tmpr,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        return None

@st.cache_data
def get_word_info(word: str, context: str) -> Optional[Dict]:
    """
    获取单词的定义、同义词、反义词和例句
    
    :param word: 要查询的单词
    :param context: 单词所在的上下文
    :return: 包含单词信息的字典或 None（如果出错）
    """
    prompt = f"""For the word "{word}" in the context of "{context}", please provide:
    1. A clear definition.
    2. Three synonyms of varying difficulty, or simple explanations if no suitable synonyms exist.
    3. An antonym (if applicable).
    4. Two example sentences using the word in different contexts.

    Please format the response as a JSON object with the following keys:
    definition, synonyms, antonym, examples
    """
    response = chat(prompt)
    if not response:
        return None
    try:
        # 移除可能的 Markdown 代码块标记
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response).strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing information for {word}: {str(e)}")
        logger.error(f"API original response: {response}")
        logger.error(f"Cleaned response: {cleaned_response}")
        return None
    except Exception as e:
        logger.error(f"Unknown error occurred while processing {word}: {str(e)}")
        return None

def process_text(text: str) -> Tuple[str, Dict]:
    """
    处理文本，查找标记的单词并获取定义
    
    :param text: 输入文本
    :return: 处理后的文本和单词信息字典的元组
    """
    words = re.findall(r'"(.*?)"', text)
    word_info = {}
    progress_bar = st.progress(0)
    for i, word in enumerate(words):
        word_info[word] = get_word_info(word, text)
        progress_bar.progress((i + 1) / len(words))
    
    # 替换文本中标记的单词
    for word in words:
        if word_info[word]:
            tooltip = f"Definition: {word_info[word]['definition']}\nSynonyms: {', '.join(word_info[word]['synonyms'])}\nAntonym: {word_info[word]['antonym']}"
            text = text.replace(f'"{word}"', f'<span class="replaced-word" title="{tooltip}">{word}</span>')
    
    return text, word_info

@st.cache_data
def get_word_definition(sentence: str) -> str:
    """
    获取句子中特定单词的英语解释和例句
    
    :param sentence: 包含要查询单词的句子
    :return: 单词的定义和例句
    """
    match = re.search(r'"(.*?)"', sentence)
    if match:
        word = match.group(1)
        prompt = f"Please provide the definition and an example sentence for the word '{word}' in the following sentence: {sentence}"
        return chat(prompt) or "Unable to get definition."
    else:
        return "No word found in quotes in the given sentence."

@st.cache_data
def analyze_text_with_gpt(paragraph: str, user_level: str) -> str:
    """
    使用 GPT 分析文本，提供难词列表和解释
    
    :param paragraph: 要分析的段落
    :param user_level: 用户的英语水平
    :return: 分析结果
    """
    prompt = f"""Analyze the following paragraph and provide:
    1. A list of difficult vocabulary for English learners at {user_level} level.
    2. For each word or phrase, provide:
       a) A definition or simpler explanation
       b) An example sentence using the word or phrase
    3. Identify any idioms or phrasal verbs and explain their meanings.

    Paragraph: {paragraph}

    Please respond in the following format:
    Difficult vocabulary:
    1. Word/Phrase: [definition/explanation] - Example: [sentence]
    2. ...

    Idiomatic expressions:
    1. Expression: [meaning] - Example: [sentence]
    2. ...
    """
    return chat(prompt) or "Unable to analyze text."

def parse_analysis_results(analysis: str) -> Tuple[List[str], List[str], List[str]]:
    """
    解析 GPT 分析结果
    
    :param analysis: GPT 分析的原始文本
    :return: 难词列表、解释列表和例句列表的元组
    """
    difficult_words = []
    explanations = []
    examples = []

    # 解析难词部分
    vocab_section = re.search(r'Difficult vocabulary:(.*?)(?:Idiomatic expressions:|$)', analysis, re.DOTALL)
    if vocab_section:
        vocab_items = re.findall(r'\d+\.\s+(.*?):\s+\[(.*?)\]\s+-\s+Example:\s+(.*?)(?=\n\d+\.|\n\n|$)', vocab_section.group(1), re.DOTALL)
        for word, explanation, example in vocab_items:
            difficult_words.append(word.strip())
            explanations.append(explanation.strip())
            examples.append(example.strip())

    # 解析习语部分
    idiom_section = re.search(r'Idiomatic expressions:(.*?)$', analysis, re.DOTALL)
    if idiom_section:
        idiom_items = re.findall(r'\d+\.\s+(.*?):\s+\[(.*?)\]\s+-\s+Example:\s+(.*?)(?=\n\d+\.|\n\n|$)', idiom_section.group(1), re.DOTALL)
        for idiom, meaning, example in idiom_items:
            difficult_words.append(idiom.strip())
            explanations.append(meaning.strip())
            examples.append(example.strip())

    return difficult_words, explanations, examples

def validate_input(text: str) -> bool:
    """
    验证输入文本
    
    :param text: 输入文本
    :return: 如果文本有效则返回 True，否则返回 False
    """
    if not text or len(text.split()) < 5:
        st.error("Please enter at least 5 words of text.")
        return False
    return True

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    获取词性标记的 WordNet POS 标记
    
    :param treebank_tag: TreeBank POS 标记
    :return: WordNet POS 标记
    """
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

def lemmatize_word(word: str) -> str:
    """
    对单词进行词形还原
    
    :param word: 要还原的单词
    :return: 词形还原后的单词
    """
    pos = nltk.pos_tag([word])[0][1]
    return lemmatizer.lemmatize(word, get_wordnet_pos(pos))

def clean_text(text: str, whitelist: Optional[List[str]] = None) -> str:
    """
    清理文本
    
    :param text: 要清理的文本
    :param whitelist: 不需要清理的单词列表
    :return: 清理后的文本
    """
    if whitelist is None:
        whitelist = []
    
    # 临时替换白名单中的单词
    for i, word in enumerate(whitelist):
        text = text.replace(word, f"WHITELIST_{i}")
    
    # 执行清理操作
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[一-龥]|[：、，。？！]|/[^/]*[\u4e00-\u9fff]+[^/]*', '', text)
    text = re.sub(r'\s*/\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 恢复白名单中的单词
    for i, word in enumerate(whitelist):
        text = text.replace(f"WHITELIST_{i}", word)
    
    return text

def replace_words_in_text(text: str, difficult_words: List[str], replacements: List[str]) -> str:
    """
    在文本中替换难词
    
    :param text: 原始文本
    :param difficult_words: 难词列表
    :param replacements: 替换词列表
    :return: 替换后的文本
    """
    whitelist = ["OPROP!"]  # 添加需要保留的特殊单词
    # 首先清理文本
    text = clean_text(text, whitelist)
    
    tokens = word_tokenize(text)
    lemmatized_difficult_words = [lemmatize_word(word.lower()) for word in difficult_words]
    
    for i, token in enumerate(tokens):
        lemmatized_token = lemmatize_word(token.lower())
        if lemmatized_token in lemmatized_difficult_words:
            index = lemmatized_difficult_words.index(lemmatized_token)
            if index < len(replacements):  # 确保索引在替换列表范围内
                replacement = replacements[index]
                tokens[i] = f'<span class="difficult-word" title="{token}: {replacement}">{replacement}</span>'
            else:
                logger.warning(f"No replacement found for '{token}'.")
    return ' '.join(tokens)

def main():
    st.title("Advanced Vocabulary Enhancement System")
    
    tab2, tab3 = st.tabs(["Word Query", "Automatic Vocabulary Replacement"])
    
   
    
    with tab2:
        st.write("Example sentence: The quick brown fox jumps over the lazy dog. \"jumps\"")
        sentence = st.text_input("Please enter an English sentence and include the word you want to query in quotes:", "The quick brown fox jumps over the lazy dog. \"jumps\"")
        if st.button("Get Definition", key="get_definition"):
            if validate_input(sentence):
                with st.spinner("Getting definition..."):
                    definition = get_word_definition(sentence)
                    st.write(f"{definition}")
    
    with tab3:
        st.write("Enter an article, and the system will automatically identify difficult vocabulary based on your English level and provide replacements and explanations.")
        user_level = st.selectbox("Choose your English level:", difficulty_levels)
        text = st.text_area("Enter your article:", height=200, key="tab3_text")
        
        if st.button("Analyze and Replace", key="analyze_and_replace"):
            if validate_input(text):
                with st.spinner("Processing..."):
                    analysis = analyze_text_with_gpt(text, user_level)
                    
                    st.subheader("Text Analysis")
                    st.markdown(analysis)
                    
                    difficult_words, explanations, examples = parse_analysis_results(analysis)

                    if len(difficult_words) != len(explanations) or len(difficult_words) != len(examples):
                        st.warning(f"Parsing mismatch: {len(difficult_words)} words, {len(explanations)} explanations, {len(examples)} examples.")

                    st.subheader("Article After Replacement")
                    replaced_text = replace_words_in_text(text, difficult_words, explanations)
                    
                    st.markdown(replaced_text, unsafe_allow_html=True)
                    
                    st.subheader("Vocabulary Explanations")
                    for word, explanation, example in zip(difficult_words, explanations, examples):
                        with st.expander(f"**{word}**"):
                            st.markdown(f"**Explanation:** {explanation}")
                            st.markdown(f"**Example:** {example}")

if __name__ == "__main__":
    main()