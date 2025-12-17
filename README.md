# Liberal AI Chatbot
社内ドキュメントを外部参照し、自然言語で問い合わせ対応ができるRAGベースのAIチャットボットです。  
社員が情報を探す時間を削減し、業務効率化を目的として開発しました。

## 概要
このアプリは RAG（Retrieval-Augmented Generation）構成を用いて、

- 会社サービスの説明

- よくあるお問い合わせ

- 社内マニュアル

- キャンペーン内容

などの情報を自然言語で回答します。

「資料を探す・担当者に聞く」といった手間をなくし、
“AIが即時に確実な根拠を持って回答してくれる仕組み” をコンセプトに設計しました。

---

## 機能

- **社内ドキュメント検索（RAG）**  
質問内容に応じて、ChromaDB から関連情報を取り出して回答します。

- **会話履歴を踏まえた回答**  
文脈を理解し、会話の流れを踏まえてスムーズに応答します。

- **回答根拠の表示（引用元パス & ページ番号）**  
どの資料を参照したかを明示し、信頼度の高い回答を実現しています。

- **ブラウザだけで動くStreamlit UI**  
社内共有しやすく、導入コストを最小限に抑えています。

- **環境変数管理（APIキーの保護）**  
dotenvによる安全なキー管理を実装しています。

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
├── components.py       # UI関係
├── initialize.py       # LLM/RAG 初期設定
├── constants.py        # 定数や設定値の管理
├── utils.py            # 共通処理
├── requirements.txt    # パッケージ一覧
└── README.md           # このドキュメント
```

---

## ▶ デモ（Streamlit Cloud）

以下のURLから、ブラウザ上でアプリを確認できます。
環境構築不要で、すぐに動作をご覧いただけます。

🔗 Demo  
https://liberal-ai-chatbot.streamlit.app/


## ローカルでの実行方法 

```bash

pip install -r requirements.txt
streamlit run app.py

```

---

## こだわったポイント

- RAGの処理をアプリ本体から分離して、構造がわかりやすくなるよう整理しました。

- 変数・設定値をconstants.pyに分離し、可読性と保守性を向上させました。

- 実際の社内利用を想定して プロンプトの振る舞いをチューニングしました。

- 会話履歴を保持したまま外部知識の参照精度が上がるようRetrieverを調整しています。

- 引用元の表示により、業務利用時の「根拠提示」ニーズに対応しています。

---

## 今後のアップデート予定

- Google Drive など外部ストレージとの連携

- ハイブリッド検索（BM25 + Embedding）の導入

---

## 作者

Shogo Ito
GitHub: https://github.com/shogo70890
