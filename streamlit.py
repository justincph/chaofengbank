# 將 streamlit 函式庫引入程式中
import streamlit as st
# 將 OpenAI 函式庫引入程式中
import openai

import re

def chat(prompt, text, tmpr, max):
    """向ChatGPT提交提示(prompt)"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": rf"Give me the function of generative dictionary, I need the definition of the word in the sentence I give you. Also, I need a sentence including the word.{prompt}"},
        ],  # 提示(promp)
        temperature=float(tmpr),
        max_tokens=int(max),
    )
    print(f"Give me the function of generative dictionary, I need the definition of the word in the sentence I give you. Also, I need a sentence including the word.")
    return response

def diff_tokens(fixed_sentence):
    '''
    切字，包含以下六種形式
    {+any char} [-any char]
    [-any char]
    {+any char}
    word\d\s
    :
    \d
    \w
    '''
    return re.findall(
        r'\{\+[^}]+?\}\s\[\-[^]]+?\]|\[\-[^]]+?\]|\{\+[^}]+?\}|[^a-zA-Z\d\s:]|:|\d+|\w+|\n',
        fixed_sentence,
    )


def get_a_line(tokens, limit=65):
    '''
    產生以65個字為一行的文章with double-space sentences
    '''
    error_html_start = ':violet[$\\tt{\\underline{'  # 顏色=violet & 並畫底線
    error_html_end = '}}$] '
    edit_html_start = ':red['  # 顏色=red
    edit_html_end = ']'

    # 追蹤當前行的長度與兩行的文本
    skip, acc_length, line1, line2, line3 = False, 0, '', '', ''

    for i, token in enumerate(tokens):
        if skip:
            skip = False
            continue
        # 若標記單位是 "{+edit+} [-error-]" 形式
        if token.startswith('{+') and token.endswith('-]'):
            edit, error = token[2:-2].split('+} [-')  # 取出word
            maxlen = max(len(error), len(edit))
            line1 += error_html_start + error + error_html_end + (' ' * (maxlen - len(error)))
            line2 += (
                edit_html_start + edit.strip() + edit_html_end + (' ' * (maxlen - len(edit) + 1))
            )
            acc_length += maxlen + 1

        # 若只有 "[-error-]" 形式
        elif token.startswith('[-'):
            error = token[2:-2]
            line1 += error_html_start + error + (' ' * (maxlen - len(error))) + error_html_end
            line2 += ' ' * (len(error) + 1)
            acc_length += len(error) + 1

        # 若只有 "{+edit+}" 形式
        elif token.startswith('{+'):
            edit = token[2:-2]
            maxlen = max(len(tokens[i + 1][1]), len(edit)) if i + 1 <= len(tokens) else len(edit)
            line1 += (' ' * (maxlen - len(tokens[i + 1][1]) + 1)) + tokens[i + 1] + ' '
            line2 += (
                edit_html_start
                + '^ '
                + edit.strip()
                + edit_html_end
                + (' ' * (maxlen - len(edit) + 1))
            )
            acc_length += len(edit) + 1
            skip = True

        # 普通單詞
        elif token.isalpha():
            line1 += token + (' ')
            line2 += ' ' * (len(token) + 1)
            acc_length += len(token) + 1

        # 標點符號
        else:
            line1 = line1.rstrip()
            line1 += token + (' ')
            line2 += ' ' * (len(token))
            acc_length += len(token) + 1

        # 檢查 acc_length 是否超過指定的行長限制
        if token == '\n' or acc_length > limit or i == len(tokens) - 1:
            print(len(line1), len(line2), len(line3))
            line1 += ' ' * (limit - len(line1))
            line2 += ' ' * (limit - len(line2))
            line3 += ' ' * (limit - len(line3))
            print(len(line1), len(line2), len(line3))

            return [line1, line2, line3], [t for t in tokens[i + 1 :]]

def replaceBlank(text):
    """處理批改後文句的空格"""
    replaced = ''
    i = 0
    while i < len(text):
        begin = text[i : i + 8]
        end = text[i : i + 4]
        while begin != ':violet[' and i < len(text):
            replaced += '&nbsp;' if text[i] == ' ' else text[i]
            i += 1
            begin = text[i : i + 8]
            continue
        while end != '}}$]' and i < len(text):
            replaced += '~' if text[i] == ' ' else text[i]
            i += 1
            end = text[i : i + 4]
            continue
    return replaced


# 將預設的提示(promp)寫出來
temp_prompt = "word"
# 也預設好要給 ChatGPT 批改的文句、temperature，及max_tokens
temp_text = "The wide range of plays and films to choose impressed me most."
temp_tmpr = 0.3
temp_max = 500


"""================================================================== streamlit 操作 =================================================================="""

# 請在下面的 API key 欄位輸入您申請的 API key。
key = st.text_input('輸入您的openai API key')
openai.api_key = key

# 設定網頁的字體
st.markdown(
    """
      <style>
        html, body, [class*="css"], p  {
        font-family: Courier New, "微軟正黑體";
        }
        [class="mord"], [class="vlist-r"] {
        font-family: Courier New, "微軟正黑體", Microsoft JhengHei;
        font-size: 16px;
        }
      </style>

      """,
    unsafe_allow_html=True,
)

# 網頁的標題
st.title("I am your English dictionary")

# 將網頁平均切分
# col1 = st.columns(2)

# 左半邊的內容
# with col1:
st.subheader("Writing Area")  # 副標題
text = ""  # 要給ChatGPT批改的文句

# 在網頁上要求使用者輸入要批改的文句，expander使得此區塊可以收起來
with st.expander("Please give your sentence", expanded=True):
    text = st.text_area("", temp_text)  # 要求使用者輸入給ChatGPT批改的文句

# 顯示批改後的文句
st.subheader('**Question box**')  # 副標題

# 若沒有輸入要批改的文句，則顯示警告"No essay"
if not text:
    st.error('No word')
# 若有，則執行以下程式碼
else:
    promptGpt = st.text_area('Write down the word you want to know in your sentence:', temp_prompt)  # 在網頁上要求使用者輸入提示(prompt)，預設為temp_prompt

    # 將網頁平均切分
    subcol1, subcol2 = st.columns(2)
    # 左半邊
    with subcol1:
        tmpr = st.text_input('Write down temperature', temp_tmpr)  # 在網頁上要求使用者輸入temperature，預設為temp_tmpr
    # 右半邊
    with subcol2:
        max_token = st.text_input('Write down # of max tokens', temp_max)  # 在網頁上要求使用者輸入max_tokens，預設為temp_max
    # 顯示目前的temperature與max_tokens
    with st.expander('Current Value'):
        st.write(rf'temperature: {tmpr}, max_tokens: {max}')

    # 讓網頁顯示送出按鈕
    submit_prompt = st.button('submit prompt')
    # 若使用者按下送出按鈕
    if submit_prompt:
        # 得到ChatGPT的回應
        response = chat(promptGpt, text, tmpr, max_token)
        # 取得批改後的文句
        fixed_sentence = response['choices'][0]['message']['content']

        # 以double space格式顯示批改後的文句
        with st.container():
            sent_tokens = diff_tokens(fixed_sentence)
            while sent_tokens:
                lines, sent_tokens = get_a_line(sent_tokens)
                line1 = replaceBlank(lines[0])
                line2 = replaceBlank(lines[1])
                st.write(line1, line2)

        # 原本的fixed_sentence
        with st.expander('Original Explanation'):
            st.write(fixed_sentence)
        # 完整的ChatGPT回應
        with st.expander('Whole Response'):
            st.write(response)

# 設定網頁的字體
st.markdown(
      """
      <style>
        html, body, [class*="css"]  {
        font-family: Courier;
        }
      </style>

      """,unsafe_allow_html=True,
      )