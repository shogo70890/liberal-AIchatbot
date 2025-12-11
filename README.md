# Liberal AI Chatbot
社内ドキュメントを外部参照し、自然言語で問い合わせ対応ができる AI チャットボットです。
社員が情報を探す時間を削減し、業務効率化を目的として開発しました。

## 概要
このアプリは、LangChain と OpenAI、ChromaDB を使って、社内ドキュメントを参照しながら回答できるRAG（Retrieval-Augmented Generation）の構成 を用いて
「会社サービスの説明」「よくある問い合わせ」「キャンペーン内容」などの情報を
自然言語で回答するチャットボットです。
社内 FAQ やサービス説明を毎回人が説明する手間をなくし、
“AIが秒速で回答してくれる仕組み” を目指して作成しました。

---

## 機能

- **社内ドキュメント検索（RAG）**  
質問内容に応じて、ChromaDB から関連情報を取り出して回答します。

- **チャット履歴の保持**  
会話の流れを踏まえてスムーズに応答できるようにしています。

- **回答の根拠となった文書のパスを表示**  
回答時に参照した文書のファイルパスとページNo.が表示されます。

- **Streamlit UI**  
ブラウザだけで動く軽量なWebアプリとして提供しています。

- **APIキーの暗号化/環境変数管理**
セキュリティを考慮した設計になっています。


---

## 技術構成
- Python 3.10  
- Streamlit  
- LangChain  
- OpenAI API  
- ChromaDB  
- dotenv / tiktoken など

---

## ファイル構成

```bash
project/
├── app.py              # Streamlit アプリ本体
├── components.py       # UIコンポーネント
├── initialize.py       # LLM/RAG 初期設定
├── constants.py        # 定数や設定値の管理
├── utils.py            # 共通処理
├── requirements.txt    # パッケージ一覧
└── README.md           # このドキュメント
```

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
