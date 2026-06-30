from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import arxiv
import requests
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = FastAPI(title="Research Search API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement des modèles (une seule fois au démarrage) ──────────────────────
print("⏳ Chargement de BART (résumé)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("⏳ Chargement de RoBERTa (question-réponse)...")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

print("✅ Modèles prêts.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_arxiv_id(url: str) -> str:
    match = re.search(r'arxiv\.org/abs/([^\s/]+)', url)
    return match.group(1) if match else ""
    
class Article(BaseModel):
    source: str
    title: str
    abstract: str
    authors: List[str]
    url: str
    year: str
    score: Optional[float] = None
    arxiv_id: Optional[str] = None
    summary: Optional[str] = None   # résumé BART
    qa_answer: Optional[str] = None # réponse RoBERTa

class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[Article]

class SearchRequest(BaseModel):
    query: str
    top_n: int = 5
    sources: str = "arxiv" # arxiv or all

class ProcessRequest(BaseModel):
    query: str
    question: str
    top_n: int = 3
    sources: str = "arxiv"

class ProcessResponse(BaseModel):
    query: str
    question: str
    total: int
    results: List[Article]

# ── Sources ───────────────────────────────────────────────────────────────────

def fetch_arxiv(query: str, max_results: int = 10) -> List[Article]:
    articles = []
    for tentative in range(3):
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for r in search.results():
                link = str(r.entry_id)
                articles.append(Article(
                    source="arXiv",
                    title=r.title,
                    abstract=r.summary,
                    authors=[a.name for a in r.authors],
                    url=link,
                    year=str(r.published.date()),
                    score=None,
                    arxiv_id=extract_arxiv_id(link)
                ))
            break
        except Exception as e:
            print(f"[arXiv] Tentative {tentative+1} échouée: {e}")
            if tentative < 2:
                time.sleep(25 + tentative * 5)
    return articles

def fetch_semantic_scholar(query: str, max_results: int = 10) -> List[Article]:
    articles = []
    for tentative in range(3):
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": max_results,
                      "fields": "title,authors,year,externalIds,abstract"}
            r = requests.get(api_url, params=params, timeout=15)
            r.raise_for_status()
            for paper in r.json().get("data", []):
                arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
                link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else \
                       f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
                abstract = paper.get("abstract") or "Résumé non disponible"
                articles.append(Article(
                    source="Semantic Scholar",
                    title=paper.get("title", "Sans titre"),
                    abstract=abstract,
                    authors=[a["name"] for a in paper.get("authors", [])],
                    url=link,
                    year=str(paper.get("year", "N/A")),
                    score=None,
                    arxiv_id=arxiv_id
                ))
            break
        except Exception as e:
            print(f"[Semantic Scholar] Tentative {tentative+1} échouée: {e}")
            if tentative < 2:
                time.sleep(20 + tentative * 5)
    return articles

def rank_by_similarity(query: str, articles: List[Article], top_n: int) -> List[Article]:
    if not articles:
        return []
    corpus = [query] + [f"{a.title} {a.abstract}" for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    for article, score in zip(articles, scores):
        article.score = round(float(score), 4)
    ranked = sorted(articles, key=lambda x: x.score or 0, reverse=True)
    filtered = [a for a in ranked if (a.score or 0) >= 0.1]
    return filtered[:top_n]

def summarize_text(text: str) -> str:
    """Résumé abstractif via BART"""
    try:
        # BART a besoin d'au moins 50 tokens
        if len(text.split()) < 50:
            return text
        result = summarizer(text, max_length=150, min_length=40, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        print(f"[BART] Erreur résumé: {e}")
        return text[:300]

def answer_question(question: str, context: str) -> str:
    """Réponse à une question via RoBERTa"""
    try:
        result = qa_model(question=question, context=context)
        return result["answer"]
    except Exception as e:
        print(f"[RoBERTa] Erreur QA: {e}")
        return "Impossible de répondre."

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ok", "message": "Research Search API v2 opérationnelle"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    articles: List[Article] = []

    if request.sources in ("arxiv", "all"):
        articles += fetch_arxiv(request.query, max_results=request.top_n)
    if request.sources in ("semantic", "all"):
        articles += fetch_semantic_scholar(request.query, max_results=request.top_n)

    seen, unique = set(), []
    for a in articles:
        key = a.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    
    ranked = rank_by_similarity(request.query, unique, request.top_n)
    return SearchResponse(query=request.query, total=len(ranked), results=ranked)

@app.post("/process", response_model=ProcessResponse)
def process(request: ProcessRequest):
    articles: List[Article] = []

    if request.sources in ("arxiv", "all"):
        articles += fetch_arxiv(request.query, max_results=10)
    if request.sources in ("semantic", "all"):
        articles += fetch_semantic_scholar(request.query, max_results=10)

    # Dédoublonner
    seen, unique = set(), []
    for a in articles:
        key = a.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    ranked = rank_by_similarity(request.query, unique, request.top_n)

    # Appliquer BART + RoBERTa sur chaque article
    for article in ranked:
        article.summary = summarize_text(article.abstract)
        article.qa_answer = answer_question(request.question, article.abstract)

    return ProcessResponse(
        query=request.query,
        question=request.question,
        total=len(ranked),
        results=ranked
    )