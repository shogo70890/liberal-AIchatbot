# Liberal AI Chatbot

友人の会社向けに作った、社内情報を自然言語で検索できるAIチャットボットです。  
LangChain と OpenAI、ChromaDB を使って、社内ドキュメントを参照しながら回答できる RAG（Retrieval-Augmented Generation）構成になっています。

---

## 機能

- **社内ドキュメント検索（RAG）**  
質問内容に応じて、ChromaDB から関連情報を取り出して回答します。

- **会話の文脈保持**  
過去のやり取りを踏まえて応答するようにしています。

- **回答の根拠となった文書のパスを表示**  
回答時に参照した文書のファイルパスとページNo.が表示されます。


---

## 技術スタック
- Python 3.10  
- Streamlit  
- LangChain  
- OpenAI API  
- ChromaDB  
- dotenv / tiktoken など

---

## ファイル構成（ざっくり）
├── app.py # Streamlit アプリ本体
├── initialize.py # VectorStore や retriever の初期化
├── components.py # UI 周り
├── utils.py # LLM 呼び出し処理
├── constants.py # プロンプト/定数
└── requirements.txt

---

## 実行方法 

```bash

pip install -r requirements.txt
streamlit run app.py

```

---

## こだわったところ

- RAGの処理をアプリ本体から分離して、構造がわかりやすくなるよう整理

- 実際に使ってもらう前提で、答えのプロンプトを調整

- 今後データソースを増やせるよう、初期化周りをモジュール化

- 回答に引用元を表示

---

## 今後やりたいこと

Google Drive など外部ストレージとの連携

---

## 作者

Shogo Ito
GitHub: https://github.com/shogo70890
