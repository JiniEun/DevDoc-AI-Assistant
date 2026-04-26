
## 💡 프로젝트

---

### 프로젝트명

👉 “DevDoc AI Assistant”

---

## 🎯 목표

> “내가 가진 문서 기반 QA 시스템 만들기”
>
> Docker로 실행되는 DevDoc AI Assistant

## 기능

- PDF / txt 업로드
- 질문 입력
- 문서 기반 답변 생성

## 구조

```java
[txt 문서]
   ↓
[HuggingFace Embedding]
   ↓
[FAISS 저장]

[질문]
   ↓
[유사 문서 검색]
   ↓
[Ollama (Llama3)]
   ↓
[답변]
```

## 목표 구성

```
[FastAPI 컨테이너]
   ├─ embedding (HF)
   ├─ FAISS
   ├─ RAG 처리
   ↓
[Ollama 컨테이너]
   ├─ llama3 모델
```

👉 핵심:

**API 서버 + LLM 서버 분리 (실무 구조)**

## 프로젝트 구조

```
devdoc-ai/
 ├── app/
 │    ├── main.py
 │    ├── rag.py
 │    ├── embedding.py
 │    └── vector_store.py
 ├── data/
 │    └── data.txt
 ├── Dockerfile
 ├── docker-compose.yml
 └── requirements.txt
```
