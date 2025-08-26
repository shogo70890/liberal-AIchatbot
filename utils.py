import logging
"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import streamlit as st

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

############################################################
# 設定関連
############################################################
# --- APIキー確認と設定 ---
key = st.secrets.get("OPENAI_API_KEY", "")
if not key.startswith("sk-"):
    st.error("OPENAI_API_KEY が正しく設定されていません。Settings → Secrets を確認してください。")
    st.stop()
os.environ["OPENAI_API_KEY"] = key  # 念のため環境変数にも反映


############################################################
# ハイブリッド検索（BM25 + Chroma）
############################################################
# ===== ここから追記：ハイブリッド検索（BM25 + Chroma） =====
from typing import List
from langchain.schema import Document
from langchain.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi

def _normalize(text: str) -> str:
    """取り込み/問い合わせで同一の正規化（最低限版）"""
    import re
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("　", " ")  # 全角空白
    return t

def _char_ngrams(s: str, n: int = 2) -> List[str]:
    s = _normalize(s)
    if len(s) <= n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

class SimpleBM25Retriever:
    """日本語向け：文字2gramでBM25。依存は rank_bm25 のみ。"""
    def __init__(self, docs: List[Document], k: int = 8, ngram: int = 2):
        self.docs = docs
        self.k = k
        self.ngram = ngram
        corpus_tokens = [_char_ngrams(d.page_content, n=self.ngram) for d in docs]
        self.bm25 = BM25Okapi(corpus_tokens)

    def get_relevant_documents(self, query: str) -> List[Document]:
        tokens = _char_ngrams(query, n=self.ngram)
        scores = self.bm25.get_scores(tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        out = []
        for i in idxs:
            d = self.docs[i].copy()
            md = d.metadata or {}
            md["_bm25_score"] = float(scores[i])
            d.metadata = md
            out.append(d)
        return out

def _doc_identity_key(d: Document):
    m = d.metadata or {}
    return (m.get("source"), m.get("page"), m.get("id"), hash(d.page_content))

def rrf_fuse(result_lists: List[List[Document]], k: int = 8, k_rrf: int = 60) -> List[Document]:
    """RRF融合。list of ranked docs を合算して上位kを返す。"""
    from collections import defaultdict
    score = defaultdict(float)
    last = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = _doc_identity_key(doc)
            last[key] = doc
            score[key] += 1.0 / (k_rrf + rank + 1)
    fused = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [last[key] for key, _ in fused]

class HybridRetriever(BaseRetriever):
    """BM25 + ベクター(MMR) をRRFで融合。LangChain互換の get_relevant_documents だけ実装。"""
    def __init__(self, bm25_ret: SimpleBM25Retriever, vec_ret, k: int = 8, k_rrf: int = 60):
        self.bm25_ret = bm25_ret
        self.vec_ret = vec_ret
        self.k = k
        self.k_rrf = k_rrf

    def _is_short(self, q: str) -> bool:
        # ざっくり：短い・名詞だけの一語などをブースト対象に
        return len(q.strip()) <= 8

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 短問のときはBM25側をやや強めに
        if self._is_short(query):
            orig_k = self.bm25_ret.k
            self.bm25_ret.k = max(orig_k, self.k + 2)
            bm25_docs = self.bm25_ret.get_relevant_documents(query)
            self.bm25_ret.k = orig_k  # 戻す
            vec_docs = self.vec_ret.get_relevant_documents(query)
        else:
            bm25_docs = self.bm25_ret.get_relevant_documents(query)
            vec_docs = self.vec_ret.get_relevant_documents(query)

        fused = rrf_fuse([bm25_docs, vec_docs], k=self.k, k_rrf=self.k_rrf)
        return fused if fused else (bm25_docs or vec_docs)

def build_hybrid_retriever(vectorstore, docs: List[Document], k: int = 8) -> BaseRetriever:
    """既存Chromaから retriever を作り、BM25 と融合して返す。"""
    vec_ret = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 80, "lambda_mult": 0.7})
    bm25_ret = SimpleBM25Retriever(docs, k=k, ngram=3)
    return HybridRetriever(bm25_ret, vec_ret, k=k, k_rrf=60)

# ===== 追記ここまで =====


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 問い合わせモードのプロンプト
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response