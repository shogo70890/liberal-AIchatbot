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
    import re
    t = text.strip().replace("　", " ")
    t = re.sub(r"\s+", " ", t)
    return t

def _char_ngrams(s: str, n: int = 2) -> List[str]:
    s = _normalize(s)
    if len(s) <= n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

class SimpleBM25:
    def __init__(self, docs: List[Document], ngram: int = 2):
        self.docs = docs
        self.ngram = ngram
        corpus = [_char_ngrams(d.page_content, n=ngram) for d in docs]
        self.bm25 = BM25Okapi(corpus)
    def topk(self, query: str, k: int) -> List[Document]:
        tokens = _char_ngrams(query, n=self.ngram)
        scores = self.bm25.get_scores(tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in idxs:
            d = self.docs[i].copy()
            md = d.metadata or {}
            md["_bm25_score"] = float(scores[i])
            d.metadata = md
            out.append(d)
        return out

def _doc_key(d: Document):
    m = d.metadata or {}
    return (m.get("source"), m.get("page"), m.get("id"), hash(d.page_content))

def rrf_fuse(ranked_lists: List[List[Document]], k: int = 8, k_rrf: int = 60) -> List[Document]:
    from collections import defaultdict
    score = defaultdict(float)
    last = {}
    for results in ranked_lists:
        for rank, doc in enumerate(results):
            key = _doc_key(doc)
            last[key] = doc
            score[key] += 1.0 / (k_rrf + rank + 1)
    fused = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [last[key] for key, _ in fused]

class HybridRetriever(BaseRetriever):
    """BM25 + Vector(MMR) をRRFで融合。LangChain準拠。"""
    def __init__(self, bm25: SimpleBM25, vec_retriever, k: int = 8, k_rrf: int = 60):
        super().__init__()
        self.bm25 = bm25
        self.vec = vec_retriever
        self.k = k
        self.k_rrf = k_rrf

    def _is_short(self, q: str) -> bool:
        return len(q.strip()) <= 8

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # BM25 / Vector それぞれ取得
        k_bm25 = self.k + 2 if self._is_short(query) else self.k
        bm25_docs = self.bm25.topk(query, k=k_bm25)
        vec_docs  = self.vec.get_relevant_documents(query)
        fused = rrf_fuse([bm25_docs, vec_docs], k=self.k, k_rrf=self.k_rrf)
        return fused if fused else (bm25_docs or vec_docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # シンプルに同期実装を使い回し
        return self._get_relevant_documents(query, run_manager=run_manager)

def build_hybrid_retriever(vectorstore, docs: List[Document], k: int = 8) -> BaseRetriever:
    vec_ret = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 60, "lambda_mult": 0.7}
    )
    bm25 = SimpleBM25(docs, ngram=2)
    return HybridRetriever(bm25=bm25, vec_retriever=vec_ret, k=k, k_rrf=60)


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