import os
import json
import base64
import uuid
import tempfile
import logging
import difflib
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Annotated, Any

from fastapi import FastAPI, APIRouter, Header, HTTPException, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, BeforeValidator
from bson import ObjectId
import httpx

from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
oai = AsyncOpenAI(api_key=OPENAI_API_KEY)
TEXT_MODEL = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4o')
IMAGE_MODEL = os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-1')
DUTCH_VOICE = os.environ.get('OPENAI_TTS_VOICE', 'nova')

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ----------------------------- Mongo helpers -----------------------------
def _validate_object_id(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)


PyObjectId = Annotated[str, BeforeValidator(_validate_object_id)]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def today_str() -> str:
    return now_utc().strftime("%Y-%m-%d")


# ----------------------------- Models -----------------------------
class SessionRequest(BaseModel):
    session_token: str


class ThemeInfo(BaseModel):
    id: str
    title: str
    subtitle: str
    icon: str
    color: str


class GenerateLessonRequest(BaseModel):
    theme_id: str
    level: str = "debutant"
    regenerate: bool = False


class GenerateStoryRequest(BaseModel):
    theme_id: Optional[str] = None
    level: str = "debutant"
    regenerate: bool = False


class TTSRequest(BaseModel):
    text: str


class FlashcardReview(BaseModel):
    card_id: str
    quality: int  # 0 hard, 1 good, 2 easy


class SaveCardsRequest(BaseModel):
    cards: List[dict]  # {dutch, french}


class CompleteLessonRequest(BaseModel):
    theme_id: str
    xp: int = 20


# ----------------------------- Theme catalog -----------------------------
THEMES: List[dict] = [
    {"id": "salutations", "title": "Salutations", "subtitle": "Dire bonjour et se présenter", "icon": "hand-wave", "color": "#FA6400"},
    {"id": "voyage", "title": "Voyage", "subtitle": "Se déplacer et demander son chemin", "icon": "airplane", "color": "#38BDF8"},
    {"id": "travail", "title": "Travail", "subtitle": "Le vocabulaire du bureau", "icon": "briefcase", "color": "#8B5CF6"},
    {"id": "nourriture", "title": "Nourriture", "subtitle": "Au restaurant et au marché", "icon": "food", "color": "#F43F5E"},
    {"id": "famille", "title": "Famille", "subtitle": "Parler de vos proches", "icon": "account-group", "color": "#34D399"},
    {"id": "shopping", "title": "Shopping", "subtitle": "Faire ses achats", "icon": "shopping", "color": "#FBBF24"},
    {"id": "nombres", "title": "Nombres & Temps", "subtitle": "Compter et donner l'heure", "icon": "numeric", "color": "#F59E0B"},
    {"id": "maison", "title": "Maison", "subtitle": "La vie à la maison", "icon": "home", "color": "#0EA5E9"},
    {"id": "sante", "title": "Santé", "subtitle": "Chez le médecin", "icon": "heart-pulse", "color": "#EF4444"},
    {"id": "expressions", "title": "Expressions", "subtitle": "Tournures idiomatiques", "icon": "chat", "color": "#EC4899"},
]

LEVEL_LABEL = {
    "debutant": "débutant (A1)",
    "intermediaire": "intermédiaire (A2-B1)",
    "avance": "avancé (B1-B2)",
    "b2": "avancé (B2)",
    "c1": "maîtrise (C1)",
}


# ----------------------------- LLM helper -----------------------------
def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # grab first { ... } or [ ... ]
    start = min([i for i in [text.find("{"), text.find("[")] if i != -1], default=0)
    return text[start:].strip()


async def llm_json(system: str, prompt: str) -> Any:
    resp = await oai.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    raw = resp.choices[0].message.content or ""
    cleaned = _strip_json(raw)
    try:
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"JSON parse failed: {e} -- raw: {raw[:400]}")
        raise HTTPException(status_code=502, detail="Génération du contenu échouée. Réessayez.")


async def generate_image(prompt: str) -> Optional[str]:
    """Generate an image via OpenAI Images, return base64 data (no prefix) or None."""
    try:
        res = await oai.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        return res.data[0].b64_json
    except Exception as e:
        logger.error(f"Image generation error: {e}")
    return None


# ----------------------------- Auth -----------------------------
async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Non authentifié")
    token = authorization.split(" ", 1)[1].strip()
    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=401, detail="Session invalide")
    expires_at = session["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < now_utc():
        raise HTTPException(status_code=401, detail="Session expirée")
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")
    return user


@api_router.post("/auth/guest")
async def create_guest():
    user_id = f"guest_{uuid.uuid4().hex[:12]}"
    await db.users.insert_one({
        "user_id": user_id,
        "email": f"{user_id}@guest.tulipe",
        "name": "Invité",
        "picture": None,
        "is_guest": True,
        "created_at": now_utc().isoformat(),
    })
    await db.progress.insert_one({
        "user_id": user_id, "xp": 0, "streak": 0, "last_active": None,
        "completed": {}, "words_learned": 0,
    })
    session_token = f"guest-{uuid.uuid4().hex}"
    await db.user_sessions.insert_one({
        "session_token": session_token,
        "user_id": user_id,
        "created_at": now_utc(),
        "expires_at": now_utc() + timedelta(days=7),
    })
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return {"session_token": session_token, "user": user}


@api_router.post("/auth/session")
async def create_session(body: SessionRequest):
    async with httpx.AsyncClient(timeout=20) as hc:
        r = await hc.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": body.session_token},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Échec de la connexion Google")
    data = r.json()
    email = data["email"]
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
        await db.users.update_one({"user_id": user_id}, {"$set": {"name": data.get("name"), "picture": data.get("picture")}})
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one({
            "user_id": user_id,
            "email": email,
            "name": data.get("name"),
            "picture": data.get("picture"),
            "created_at": now_utc().isoformat(),
        })
        await db.progress.insert_one({
            "user_id": user_id, "xp": 0, "streak": 0, "last_active": None,
            "completed": {}, "words_learned": 0,
        })

    session_token = data["session_token"]
    await db.user_sessions.update_one(
        {"session_token": session_token},
        {"$set": {
            "session_token": session_token,
            "user_id": user_id,
            "created_at": now_utc(),
            "expires_at": now_utc() + timedelta(days=7),
        }},
        upsert=True,
    )
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return {"session_token": session_token, "user": user}


from fastapi import Depends


@api_router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return user


@api_router.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        await db.user_sessions.delete_one({"session_token": token})
    return {"ok": True}


# ----------------------------- Themes -----------------------------
@api_router.get("/themes")
async def get_themes():
    return THEMES


# ----------------------------- Lessons -----------------------------
@api_router.post("/lessons/generate")
async def generate_lesson(body: GenerateLessonRequest, user: dict = Depends(get_current_user)):
    theme = next((t for t in THEMES if t["id"] == body.theme_id), None)
    if not theme:
        raise HTTPException(status_code=404, detail="Thème introuvable")
    cache_key = f"lesson::{body.theme_id}::{body.level}::v2"
    if not body.regenerate:
        cached = await db.generated_content.find_one({"key": cache_key}, {"_id": 0})
        if cached:
            return cached["content"]

    level = LEVEL_LABEL.get(body.level, "débutant (A1)")
    system = (
        "Tu es un professeur de néerlandais pour francophones. "
        "Tu réponds UNIQUEMENT avec du JSON valide, sans texte autour, sans markdown."
    )
    prompt = f"""Crée une leçon de néerlandais sur le thème « {theme['title']} » ({theme['subtitle']}) pour un niveau {level}.
Renvoie un objet JSON avec EXACTEMENT cette structure:
{{
  "theme_id": "{body.theme_id}",
  "title": "titre court en français",
  "intro": "1 phrase de contexte en français",
  "vocabulary": [
    {{"dutch": "mot néerlandais", "french": "traduction française", "phon": "transcription phonétique API (IPA) du mot, sans crochets", "phon_fr": "prononciation approximative écrite à la française", "example_nl": "phrase exemple en néerlandais", "example_fr": "traduction de la phrase"}}
  ],
  "exercises": [
    {{"type": "mcq", "question": "Comment dit-on ... en néerlandais ?", "prompt_nl": "", "options": ["a","b","c","d"], "answer": 0, "explanation": "courte explication en français"}},
    {{"type": "translate", "question": "Traduisez en néerlandais : ...", "prompt_fr": "phrase française", "answer": "réponse néerlandaise attendue", "explanation": "explication"}}
  ]
}}
Contraintes: 8 mots de vocabulaire pertinents, 6 exercices variés (4 de type "mcq", 2 de type "translate"). Pour les mcq, "answer" est l'index (0-3) de la bonne option. Tout le méta-texte est en français, le vocabulaire cible en néerlandais."""

    content = await llm_json(system, prompt)
    content["theme_id"] = body.theme_id
    await db.generated_content.update_one(
        {"key": cache_key}, {"$set": {"key": cache_key, "content": content, "created_at": now_utc().isoformat()}}, upsert=True
    )
    return content


# ----------------------------- Grammar / Writing -----------------------------
@api_router.post("/grammar/generate")
async def generate_grammar(body: GenerateLessonRequest, user: dict = Depends(get_current_user)):
    cache_key = f"grammar::{body.theme_id}::{body.level}"
    if not body.regenerate:
        cached = await db.generated_content.find_one({"key": cache_key}, {"_id": 0})
        if cached:
            return cached["content"]
    level = LEVEL_LABEL.get(body.level, "débutant (A1)")
    system = "Tu es un professeur de grammaire néerlandaise pour francophones. Réponds UNIQUEMENT en JSON valide."
    prompt = f"""Crée une fiche de grammaire néerlandaise de niveau {level}.
JSON:
{{
  "title": "titre de la règle en français",
  "rule": "explication claire de la règle en français (3-4 phrases)",
  "examples": [{{"nl": "phrase néerlandaise", "fr": "traduction"}}],
  "exercises": [{{"type": "fill", "question": "consigne en français", "sentence": "phrase avec ___", "answer": "mot manquant", "explanation": "explication"}}]
}}
3 exemples, 5 exercices de type "fill" (texte à trous)."""
    content = await llm_json(system, prompt)
    await db.generated_content.update_one(
        {"key": cache_key}, {"$set": {"key": cache_key, "content": content, "created_at": now_utc().isoformat()}}, upsert=True
    )
    return content


# ----------------------------- Stories -----------------------------
@api_router.post("/story/generate")
async def generate_story(body: GenerateStoryRequest, user: dict = Depends(get_current_user)):
    theme = next((t for t in THEMES if t["id"] == body.theme_id), None) if body.theme_id else None
    theme_title = theme["title"] if theme else "la vie quotidienne"
    cache_key = f"story::{body.theme_id or 'general'}::{body.level}::v2"
    if not body.regenerate:
        cached = await db.generated_content.find_one({"key": cache_key}, {"_id": 0})
        if cached:
            return cached["content"]
    level = LEVEL_LABEL.get(body.level, "débutant (A1)")
    system = "Tu es un auteur d'histoires courtes en néerlandais pour apprenants francophones. Réponds UNIQUEMENT en JSON valide."
    prompt = f"""Écris une histoire courte en néerlandais sur « {theme_title} » pour un niveau {level}.
JSON:
{{
  "id": "{cache_key}",
  "title_nl": "titre en néerlandais",
  "title_fr": "titre en français",
  "sentences": [{{"nl": "phrase en néerlandais", "fr": "traduction française", "phon": "transcription phonétique API (IPA) de la phrase, sans crochets"}}],
  "keywords": [{{"dutch": "mot clé", "french": "traduction", "phon": "transcription IPA du mot, sans crochets"}}]
}}
8 à 10 phrases formant une histoire cohérente, 6 mots-clés. Phrases adaptées au niveau."""
    content = await llm_json(system, prompt)
    content["id"] = cache_key
    if theme:
        content["theme_title"] = theme["title"]
    await db.generated_content.update_one(
        {"key": cache_key}, {"$set": {"key": cache_key, "content": content, "created_at": now_utc().isoformat()}}, upsert=True
    )
    return content


# ----------------------------- Exercises (workbook) -----------------------------
EXERCISE_CATEGORIES = [
    {"id": "conjugaison", "title": "Conjugaison", "subtitle": "Conjugue les verbes", "icon": "sync", "color": "#FA6400",
     "guidance": "exercices de conjugaison de verbes néerlandais (réguliers, irréguliers, zijn/hebben) : surtout des exercices de type 'fill' avec une phrase à trous où l'élève doit écrire la forme conjugée correcte"},
    {"id": "temps", "title": "Temps", "subtitle": "Présent, passé, futur", "icon": "clock-outline", "color": "#38BDF8",
     "guidance": "exercices sur les temps verbaux (tegenwoordige tijd, voltooid tegenwoordige tijd/perfectum, verleden tijd, futur avec 'zullen') : transformer ou compléter des phrases, mélange de 'fill' et 'mcq'"},
    {"id": "grammaire", "title": "Grammaire", "subtitle": "Règles et structures", "icon": "book-alphabet", "color": "#8B5CF6",
     "guidance": "exercices de grammaire néerlandaise (ordre des mots, articles de/het, pronoms, prépositions, négation) : surtout des 'mcq' où l'élève choisit la forme correcte"},
    {"id": "orthographe", "title": "Orthographe", "subtitle": "Écris sans fautes", "icon": "spellcheck", "color": "#34D399",
     "guidance": "exercices d'orthographe néerlandaise (doublement de consonnes/voyelles, règle 't kofschip', pluriels) : 'mcq' pour choisir la bonne orthographe ou 'fill' pour écrire correctement le mot"},
    {"id": "vocabulaire", "title": "Vocabulaire", "subtitle": "Enrichis ton lexique", "icon": "translate", "color": "#F43F5E",
     "guidance": "exercices de vocabulaire : traductions FR<->NL, synonymes, mots dans leur contexte, mélange de 'mcq' et 'fill'"},
]


@api_router.get("/exercises/categories")
async def exercises_categories(user: dict = Depends(get_current_user)):
    return [{k: v for k, v in c.items() if k != "guidance"} for c in EXERCISE_CATEGORIES]


@api_router.post("/exercises/generate")
async def generate_exercises(body: GenerateLessonRequest, user: dict = Depends(get_current_user)):
    cat = next((c for c in EXERCISE_CATEGORIES if c["id"] == body.theme_id), None)
    if not cat:
        raise HTTPException(status_code=404, detail="Catégorie introuvable")
    cache_key = f"exo::{body.theme_id}::{body.level}"
    if not body.regenerate:
        cached = await db.generated_content.find_one({"key": cache_key}, {"_id": 0})
        if cached:
            return cached["content"]
    level = LEVEL_LABEL.get(body.level, "débutant (A1)")
    system = "Tu es un professeur de néerlandais qui crée des exercices de cahier pour francophones. Réponds UNIQUEMENT en JSON valide, sans markdown."
    prompt = f"""Crée une fiche d'exercices de néerlandais, catégorie « {cat['title']} », niveau {level}.
Type d'exercices attendus : {cat['guidance']}.
Renvoie ce JSON:
{{
  "category": "{body.theme_id}",
  "title": "titre court en français",
  "exercises": [
    {{"type": "mcq", "question": "consigne/question en français", "options": ["a","b","c","d"], "answer": 0, "explanation": "explication courte en français"}},
    {{"type": "fill", "question": "consigne en français", "sentence": "phrase néerlandaise avec ___", "answer": "réponse attendue", "explanation": "explication en français"}}
  ]
}}
Contraintes : 8 exercices adaptés au niveau et à la catégorie. Pour "mcq", "answer" est l'index (0-3) de la bonne option. Toutes les consignes/explications en français, le contenu cible en néerlandais."""
    content = await llm_json(system, prompt)
    content["category"] = body.theme_id
    await db.generated_content.update_one(
        {"key": cache_key}, {"$set": {"key": cache_key, "content": content, "created_at": now_utc().isoformat()}}, upsert=True
    )
    return content


@api_router.get("/leaderboard")
async def leaderboard(user: dict = Depends(get_current_user)):
    rows = await db.progress.find({"xp": {"$gt": 0}}, {"_id": 0, "user_id": 1, "xp": 1}).sort("xp", -1).to_list(300)
    ids = [r["user_id"] for r in rows]
    users = {}
    if ids:
        async for u in db.users.find({"user_id": {"$in": ids}}, {"_id": 0, "user_id": 1, "name": 1, "picture": 1}):
            users[u["user_id"]] = u
    ranked = []
    for i, r in enumerate(rows):
        u = users.get(r["user_id"], {})
        ranked.append({
            "rank": i + 1,
            "user_id": r["user_id"],
            "name": u.get("name") or "Anonyme",
            "picture": u.get("picture"),
            "xp": r["xp"],
            "is_me": r["user_id"] == user["user_id"],
        })
    top = ranked[:50]
    me = next((r for r in ranked if r["is_me"]), None)
    if me is None:
        myprog = await db.progress.find_one({"user_id": user["user_id"]}, {"_id": 0, "xp": 1})
        me = {
            "rank": len(ranked) + 1,
            "user_id": user["user_id"],
            "name": user.get("name") or "Toi",
            "picture": user.get("picture"),
            "xp": (myprog or {}).get("xp", 0),
            "is_me": True,
        }
    return {"top": top, "me": me, "total_players": len(ranked)}


# ----------------------------- Phonetics -----------------------------
class PhoneticsRequest(BaseModel):
    texts: List[str]


@api_router.post("/phonetics")
async def phonetics(body: PhoneticsRequest, user: dict = Depends(get_current_user)):
    out = []
    to_generate = []
    for t in body.texts:
        key = (t or "").strip().lower()
        if not key:
            continue
        cached = await db.phonetics.find_one({"key": key}, {"_id": 0})
        if cached:
            out.append({"text": t, "phon": cached.get("phon", ""), "phon_fr": cached.get("phon_fr", "")})
        else:
            to_generate.append(t)
    if to_generate:
        system = "Tu es phonéticien spécialiste du néerlandais. Réponds UNIQUEMENT en JSON valide, sans markdown."
        prompt = (
            "Pour chaque texte néerlandais, donne sa transcription phonétique en Alphabet Phonétique International (IPA) "
            "et une prononciation approximative écrite « à la française ». "
            f"Textes: {json.dumps(to_generate, ensure_ascii=False)}. "
            'Renvoie ce JSON: {"results": [{"text": "le texte", "phon": "transcription IPA sans crochets", "phon_fr": "prononciation approximative française"}]}'
        )
        try:
            data = await llm_json(system, prompt)
            for r in data.get("results", []):
                key = (r.get("text") or "").strip().lower()
                if not key:
                    continue
                doc = {"key": key, "phon": r.get("phon", ""), "phon_fr": r.get("phon_fr", "")}
                await db.phonetics.update_one({"key": key}, {"$set": doc}, upsert=True)
                out.append({"text": r.get("text"), "phon": doc["phon"], "phon_fr": doc["phon_fr"]})
        except Exception as e:
            logger.error(f"Phonetics error: {e}")
    return {"results": out}


class WordInfoRequest(BaseModel):
    text: str


@api_router.post("/word-info")
async def word_info(body: WordInfoRequest, user: dict = Depends(get_current_user)):
    key = (body.text or "").strip().lower()
    if not key:
        raise HTTPException(status_code=400, detail="Mot vide")
    cached = await db.word_info.find_one({"key": key}, {"_id": 0})
    if cached:
        return {"text": body.text, "french": cached.get("french", ""), "phon": cached.get("phon", ""), "phon_fr": cached.get("phon_fr", "")}
    system = "Tu es professeur de néerlandais pour francophones. Réponds UNIQUEMENT en JSON valide, sans markdown."
    prompt = (
        f'Pour le mot ou la courte expression néerlandaise "{body.text}", donne sa traduction française, '
        "sa transcription phonétique IPA (sans crochets) et une prononciation approximative « à la française ». "
        'Renvoie ce JSON: {"french": "traduction", "phon": "IPA sans crochets", "phon_fr": "approx française"}'
    )
    data = await llm_json(system, prompt)
    doc = {"key": key, "french": data.get("french", ""), "phon": data.get("phon", ""), "phon_fr": data.get("phon_fr", "")}
    await db.word_info.update_one({"key": key}, {"$set": doc}, upsert=True)
    return {"text": body.text, "french": doc["french"], "phon": doc["phon"], "phon_fr": doc["phon_fr"]}


# ----------------------------- TTS -----------------------------
@api_router.post("/tts")
async def tts(body: TTSRequest, user: dict = Depends(get_current_user)):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide")
    try:
        speech = await oai.audio.speech.create(
            model="tts-1", voice=DUTCH_VOICE, input=body.text[:800], response_format="mp3"
        )
        b64 = base64.b64encode(speech.content).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=502, detail="Synthèse vocale indisponible")
    return {"audio_base64": b64, "format": "mp3"}


# ----------------------------- Pronunciation -----------------------------
def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zàâäéèêëïîôöùûüçñ0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@api_router.post("/pronunciation/evaluate")
async def pronunciation_evaluate(
    target: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    suffix = Path(file.filename or "audio.m4a").suffix or ".m4a"
    if suffix.lstrip(".").lower() not in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
        suffix = ".m4a"
    content = await file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as fh:
            result = await oai.audio.transcriptions.create(
                file=fh, model="whisper-1", response_format="json", language="nl"
            )
        transcript = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else str(result))
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=502, detail="Analyse vocale indisponible")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    ratio = difflib.SequenceMatcher(None, _normalize(target), _normalize(transcript or "")).ratio()
    score = round(ratio * 100)
    if score >= 85:
        feedback = "Excellent ! Votre prononciation est très claire."
    elif score >= 65:
        feedback = "Bien ! Quelques sons à peaufiner, réécoutez le modèle."
    elif score >= 40:
        feedback = "Pas mal. Ralentissez et articulez chaque mot."
    else:
        feedback = "Réessayez en parlant plus fort et plus près du micro."
    return {"transcript": transcript, "score": score, "feedback": feedback, "target": target}


# ----------------------------- Progress -----------------------------
async def award_xp(user_id: str, amount: int, theme_id: Optional[str] = None) -> dict:
    prog = await db.progress.find_one({"user_id": user_id}, {"_id": 0}) or {
        "user_id": user_id, "xp": 0, "streak": 0, "last_active": None, "completed": {}, "words_learned": 0
    }
    today = today_str()
    last = prog.get("last_active")
    streak = prog.get("streak", 0)
    if last == today:
        pass
    elif last == (now_utc() - timedelta(days=1)).strftime("%Y-%m-%d"):
        streak += 1
    else:
        streak = 1
    daily = prog.get("daily_xp") or {}
    today_xp = (daily.get("xp", 0) if daily.get("date") == today else 0) + amount
    xp = prog.get("xp", 0) + amount
    upd = {"xp": xp, "streak": streak, "last_active": today, "daily_xp": {"date": today, "xp": today_xp}}
    completed = prog.get("completed", {})
    if theme_id:
        completed[theme_id] = completed.get(theme_id, 0) + 1
        upd["completed"] = completed
    await db.progress.update_one({"user_id": user_id}, {"$set": upd}, upsert=True)
    return {"xp": xp, "streak": streak, "today_xp": today_xp, "completed": completed}


@api_router.get("/progress")
async def get_progress(user: dict = Depends(get_current_user)):
    prog = await db.progress.find_one({"user_id": user["user_id"]}, {"_id": 0})
    if not prog:
        prog = {"user_id": user["user_id"], "xp": 0, "streak": 0, "last_active": None, "completed": {}, "words_learned": 0}
        await db.progress.insert_one(dict(prog))
    due = await db.flashcards.count_documents({"user_id": user["user_id"], "due_date": {"$lte": today_str()}})
    total_cards = await db.flashcards.count_documents({"user_id": user["user_id"]})
    daily = prog.get("daily_xp") or {}
    prog["today_xp"] = daily.get("xp", 0) if daily.get("date") == today_str() else 0
    prog["daily_goal"] = prog.get("daily_goal", 30)
    prog["due_cards"] = due
    prog["total_cards"] = total_cards
    return prog


class GoalRequest(BaseModel):
    goal: int


@api_router.post("/progress/goal")
async def set_goal(body: GoalRequest, user: dict = Depends(get_current_user)):
    goal = max(10, min(200, body.goal))
    await db.progress.update_one({"user_id": user["user_id"]}, {"$set": {"daily_goal": goal}}, upsert=True)
    return {"daily_goal": goal}


@api_router.post("/progress/complete")
async def complete_lesson(body: CompleteLessonRequest, user: dict = Depends(get_current_user)):
    res = await award_xp(user["user_id"], body.xp, body.theme_id)
    return res


# ----------------------------- Flashcards (SRS) -----------------------------
@api_router.post("/flashcards/save")
async def save_cards(body: SaveCardsRequest, user: dict = Depends(get_current_user)):
    saved = 0
    for c in body.cards:
        dutch = (c.get("dutch") or "").strip()
        french = (c.get("french") or "").strip()
        if not dutch:
            continue
        exists = await db.flashcards.find_one({"user_id": user["user_id"], "dutch": dutch})
        if exists:
            continue
        await db.flashcards.insert_one({
            "card_id": f"card_{uuid.uuid4().hex[:12]}",
            "user_id": user["user_id"],
            "dutch": dutch,
            "french": french,
            "phon": (c.get("phon") or "").strip(),
            "interval": 1,
            "ease": 2.5,
            "reps": 0,
            "due_date": today_str(),
            "created_at": now_utc().isoformat(),
        })
        saved += 1
    if saved:
        await db.progress.update_one({"user_id": user["user_id"]}, {"$inc": {"words_learned": saved}}, upsert=True)
    return {"saved": saved}


@api_router.get("/flashcards/due")
async def get_due_cards(user: dict = Depends(get_current_user)):
    cards = await db.flashcards.find(
        {"user_id": user["user_id"], "due_date": {"$lte": today_str()}}, {"_id": 0}
    ).limit(30).to_list(30)
    return cards


@api_router.get("/flashcards/all")
async def get_all_cards(user: dict = Depends(get_current_user)):
    cards = await db.flashcards.find({"user_id": user["user_id"]}, {"_id": 0}).to_list(500)
    return cards


@api_router.post("/flashcards/review")
async def review_card(body: FlashcardReview, user: dict = Depends(get_current_user)):
    card = await db.flashcards.find_one({"card_id": body.card_id, "user_id": user["user_id"]}, {"_id": 0})
    if not card:
        raise HTTPException(status_code=404, detail="Carte introuvable")
    interval = card.get("interval", 1)
    reps = card.get("reps", 0) + 1
    if body.quality == 0:
        interval = 1
    elif body.quality == 1:
        interval = max(2, round(interval * 2))
    else:
        interval = max(4, round(interval * 2.5))
    due = (now_utc() + timedelta(days=interval)).strftime("%Y-%m-%d")
    await db.flashcards.update_one(
        {"card_id": body.card_id},
        {"$set": {"interval": interval, "reps": reps, "due_date": due}},
    )
    return {"card_id": body.card_id, "next_due": due, "interval": interval}


# ----------------------------- Conversation (AI dialogue partner) -----------------------------
class ConversationStart(BaseModel):
    scenario_id: str
    level: str = "debutant"


class ConversationMessage(BaseModel):
    conversation_id: str
    text: str


SCENARIOS = [
    {"id": "presentation", "title": "Se présenter", "subtitle": "Nom, âge, profession", "icon": "hand-wave", "color": "#FA6400", "context": "vous faites connaissance et vous présentez", "image_prompt": "a friendly candid photo of two people meeting and shaking hands in a bright Dutch living room, natural light, realistic"},
    {"id": "cafe", "title": "Au café", "subtitle": "Commander une boisson", "icon": "coffee", "color": "#F43F5E", "context": "l'élève commande dans un café aux Pays-Bas", "image_prompt": "a cozy Dutch café interior with coffee cups, pastries and a menu on a wooden table, warm lighting, realistic photo"},
    {"id": "gare", "title": "À la gare", "subtitle": "Acheter un billet, demander l'heure", "icon": "train", "color": "#38BDF8", "context": "l'élève achète un billet de train et demande des informations", "image_prompt": "a modern Dutch train station platform with a yellow NS train, departure boards, realistic photo"},
    {"id": "courses", "title": "Faire les courses", "subtitle": "Au supermarché / marché", "icon": "cart", "color": "#34D399", "context": "l'élève fait ses courses au marché", "image_prompt": "a colorful Dutch outdoor market stall with fruits, vegetables, cheese and flowers, realistic photo"},
    {"id": "medecin", "title": "Chez le médecin", "subtitle": "Expliquer un symptôme", "icon": "stethoscope", "color": "#EF4444", "context": "l'élève explique ses symptômes chez le médecin", "image_prompt": "a bright friendly doctor's office in the Netherlands with a desk and medical items, realistic photo"},
    {"id": "libre", "title": "Conversation libre", "subtitle": "Parle de tout", "icon": "chat", "color": "#8B5CF6", "context": "conversation libre du quotidien", "image_prompt": "a beautiful everyday scene of a Dutch city street with canal houses and bicycles, realistic photo"},
]


def _tutor_system(scenario_ctx: str, level: str) -> str:
    lvl = LEVEL_LABEL.get(level, "débutant (A1)")
    return (
        f"Tu es Anke, une professeure de néerlandais chaleureuse et patiente. Tu discutes avec un élève "
        f"francophone de niveau {lvl} pour l'aider à pratiquer le néerlandais à l'oral et à l'écrit. "
        f"Contexte du dialogue : {scenario_ctx}. "
        "Règles : garde tes répliques COURTES (1-2 phrases) et naturelles, en néerlandais adapté au niveau. "
        "Pose souvent une question pour relancer la conversation. Corrige gentiment les erreurs de l'élève "
        "et explique en français de manière détaillée mais concise. Réponds UNIQUEMENT en JSON valide, sans markdown."
    )


@api_router.get("/conversation/scenarios")
async def conversation_scenarios(user: dict = Depends(get_current_user)):
    return [{k: v for k, v in s.items() if k not in ("context", "image_prompt")} for s in SCENARIOS]


@api_router.post("/conversation/start")
async def conversation_start(body: ConversationStart, user: dict = Depends(get_current_user)):
    scenario = next((s for s in SCENARIOS if s["id"] == body.scenario_id), None)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scénario introuvable")
    conv_id = f"conv_{uuid.uuid4().hex[:12]}"
    system = _tutor_system(scenario["context"], body.level)
    prompt = (
        "Commence la conversation par une première réplique d'accueil adaptée au scénario. "
        'Renvoie ce JSON: {"reply_nl": "réplique en néerlandais", "reply_fr": "traduction française", '
        '"phon": "transcription IPA de reply_nl sans crochets", '
        '"words": [{"nl": "chaque mot néerlandais de la réplique dans l\'ordre", "fr": "sa traduction française", "phon": "IPA du mot sans crochets"}], '
        '"tip": "court conseil ou explication en français pour aider l\'élève à répondre"}'
    )
    data = await llm_json(system, prompt)
    image_b64 = await generate_image(scenario["image_prompt"])
    now = now_utc().isoformat()
    await db.conversations.insert_one({
        "conv_id": conv_id,
        "user_id": user["user_id"],
        "scenario_id": body.scenario_id,
        "level": body.level,
        "title": scenario["title"],
        "created_at": now,
    })
    msg = {
        "conv_id": conv_id,
        "user_id": user["user_id"],
        "role": "tutor",
        "dutch": data.get("reply_nl", ""),
        "french": data.get("reply_fr", ""),
        "phon": data.get("phon", ""),
        "words": data.get("words", []),
        "correction": "",
        "tip": data.get("tip", ""),
        "image": image_b64,
        "order": 0,
        "created_at": now,
    }
    await db.conversation_messages.insert_one(dict(msg))
    msg.pop("_id", None)
    return {"conversation_id": conv_id, "scenario": {k: v for k, v in scenario.items() if k not in ("context", "image_prompt")}, "message": msg}


@api_router.post("/conversation/message")
async def conversation_send(body: ConversationMessage, user: dict = Depends(get_current_user)):
    conv = await db.conversations.find_one({"conv_id": body.conversation_id, "user_id": user["user_id"]}, {"_id": 0})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    scenario = next((s for s in SCENARIOS if s["id"] == conv["scenario_id"]), None)
    ctx = scenario["context"] if scenario else "conversation du quotidien"

    prior = await db.conversation_messages.find({"conv_id": body.conversation_id}, {"_id": 0}).sort("order", 1).to_list(200)
    count = len(prior)

    # store user message
    now = now_utc().isoformat()
    await db.conversation_messages.insert_one({
        "conv_id": body.conversation_id,
        "user_id": user["user_id"],
        "role": "user",
        "text": body.text,
        "order": count,
        "created_at": now,
    })

    transcript_lines = []
    for m in prior:
        if m["role"] == "tutor":
            transcript_lines.append(f"Anke: {m.get('dutch','')}")
        else:
            transcript_lines.append(f"Élève: {m.get('text','')}")
    transcript_lines.append(f"Élève: {body.text}")
    transcript = "\n".join(transcript_lines)

    system = _tutor_system(ctx, conv.get("level", "debutant"))
    prompt = (
        f"Voici la conversation jusqu'ici :\n{transcript}\n\n"
        "Réponds à la dernière réplique de l'élève. "
        'Renvoie ce JSON: {"reply_nl": "ta réponse en néerlandais", "reply_fr": "traduction française", '
        '"phon": "transcription IPA de reply_nl sans crochets", '
        '"words": [{"nl": "chaque mot néerlandais de ta réponse dans l\'ordre", "fr": "sa traduction française", "phon": "IPA du mot sans crochets"}], '
        '"correction": "correction en français de la dernière phrase de l\'élève si elle contient des erreurs, sinon chaîne vide", '
        '"tip": "courte explication de grammaire ou vocabulaire en français liée à l\'échange"}'
    )
    data = await llm_json(system, prompt)
    now2 = now_utc().isoformat()
    tutor_msg = {
        "conv_id": body.conversation_id,
        "user_id": user["user_id"],
        "role": "tutor",
        "dutch": data.get("reply_nl", ""),
        "french": data.get("reply_fr", ""),
        "phon": data.get("phon", ""),
        "words": data.get("words", []),
        "correction": data.get("correction", "") or "",
        "tip": data.get("tip", ""),
        "image": None,
        "order": count + 1,
        "created_at": now2,
    }
    await db.conversation_messages.insert_one(dict(tutor_msg))
    tutor_msg.pop("_id", None)
    return {"message": tutor_msg}


@api_router.get("/conversations")
async def conversations_list(user: dict = Depends(get_current_user)):
    convs = await db.conversations.find({"user_id": user["user_id"]}, {"_id": 0}).sort("created_at", -1).to_list(100)
    out = []
    for c in convs:
        last = await db.conversation_messages.find(
            {"conv_id": c["conv_id"]}, {"_id": 0, "dutch": 1, "text": 1, "role": 1, "order": 1}
        ).sort("order", -1).limit(1).to_list(1)
        preview = ""
        if last:
            lm = last[0]
            preview = lm.get("dutch") or lm.get("text") or ""
        count = await db.conversation_messages.count_documents({"conv_id": c["conv_id"]})
        scenario = next((s for s in SCENARIOS if s["id"] == c.get("scenario_id")), None)
        out.append({
            "conv_id": c["conv_id"],
            "title": c.get("title", "Conversation"),
            "scenario_id": c.get("scenario_id"),
            "icon": scenario["icon"] if scenario else "chat",
            "color": scenario["color"] if scenario else "#8B5CF6",
            "level": c.get("level"),
            "created_at": c.get("created_at"),
            "preview": preview,
            "message_count": count,
        })
    return out


@api_router.post("/conversation/photo")
async def conversation_photo(body: ConversationMessage, user: dict = Depends(get_current_user)):
    """Anke shares a new photo and comments on it (like a real person)."""
    conv = await db.conversations.find_one({"conv_id": body.conversation_id, "user_id": user["user_id"]}, {"_id": 0})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    scenario = next((s for s in SCENARIOS if s["id"] == conv["scenario_id"]), None)
    ctx = scenario["context"] if scenario else "conversation du quotidien"
    img_prompt = scenario["image_prompt"] if scenario else "a realistic everyday photo in the Netherlands"

    image_b64 = await generate_image(img_prompt)
    prior = await db.conversation_messages.find({"conv_id": body.conversation_id}, {"_id": 0}).sort("order", 1).to_list(200)
    count = len(prior)

    system = _tutor_system(ctx, conv.get("level", "debutant"))
    prompt = (
        f"Tu viens de partager avec l'élève une photo montrant : {img_prompt}. "
        "Décris brièvement la photo en néerlandais (1-2 phrases) et pose une question à l'élève à propos de la photo. "
        'Renvoie ce JSON: {"reply_nl": "description + question en néerlandais", "reply_fr": "traduction française", '
        '"phon": "transcription IPA de reply_nl sans crochets", '
        '"words": [{"nl": "chaque mot néerlandais dans l\'ordre", "fr": "sa traduction", "phon": "IPA du mot sans crochets"}], '
        '"tip": "courte explication utile en français"}'
    )
    data = await llm_json(system, prompt)
    now = now_utc().isoformat()
    tutor_msg = {
        "conv_id": body.conversation_id,
        "user_id": user["user_id"],
        "role": "tutor",
        "dutch": data.get("reply_nl", ""),
        "french": data.get("reply_fr", ""),
        "phon": data.get("phon", ""),
        "words": data.get("words", []),
        "correction": "",
        "tip": data.get("tip", ""),
        "image": image_b64,
        "order": count,
        "created_at": now,
    }
    await db.conversation_messages.insert_one(dict(tutor_msg))
    tutor_msg.pop("_id", None)
    return {"message": tutor_msg}


@api_router.get("/conversation/{conv_id}")
async def conversation_get(conv_id: str, user: dict = Depends(get_current_user)):
    conv = await db.conversations.find_one({"conv_id": conv_id, "user_id": user["user_id"]}, {"_id": 0})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    msgs = await db.conversation_messages.find({"conv_id": conv_id}, {"_id": 0}).sort("order", 1).to_list(500)
    return {"conversation": conv, "messages": msgs}


@api_router.post("/conversation/end")
async def conversation_end(body: ConversationMessage, user: dict = Depends(get_current_user)):
    conv = await db.conversations.find_one({"conv_id": body.conversation_id, "user_id": user["user_id"]}, {"_id": 0})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    msgs = await db.conversation_messages.find({"conv_id": body.conversation_id}, {"_id": 0}).sort("order", 1).to_list(500)
    lines = []
    for m in msgs:
        if m["role"] == "tutor":
            lines.append(f"Anke: {m.get('dutch','')}")
        else:
            lines.append(f"Élève: {m.get('text','')}")
    transcript = "\n".join(lines) or "(conversation vide)"
    system = "Tu es Anke, professeure de néerlandais. Tu fais un bilan bienveillant à un élève francophone. Réponds UNIQUEMENT en JSON valide."
    prompt = (
        f"Voici la conversation :\n{transcript}\n\n"
        "Fais un bilan court et motivant en français. "
        'Renvoie ce JSON: {"summary": "résumé en 1-2 phrases de ce qui a été pratiqué", '
        '"strengths": "1 point fort en français", "improvements": "1 conseil d\'amélioration en français", '
        '"words": [{"dutch": "mot ou expression clé utilisé", "french": "traduction"}]}. '
        "Mets 5 à 8 mots-clés utiles vus dans la conversation."
    )
    data = await llm_json(system, prompt)

    saved = 0
    for c in (data.get("words") or []):
        dutch = (c.get("dutch") or "").strip()
        french = (c.get("french") or "").strip()
        if not dutch:
            continue
        exists = await db.flashcards.find_one({"user_id": user["user_id"], "dutch": dutch})
        if exists:
            continue
        await db.flashcards.insert_one({
            "card_id": f"card_{uuid.uuid4().hex[:12]}",
            "user_id": user["user_id"],
            "dutch": dutch,
            "french": french,
            "interval": 1, "ease": 2.5, "reps": 0,
            "due_date": today_str(),
            "created_at": now_utc().isoformat(),
        })
        saved += 1
    if saved:
        await db.progress.update_one({"user_id": user["user_id"]}, {"$inc": {"words_learned": saved}}, upsert=True)

    xp_amount = 15 + saved * 2
    prog = await award_xp(user["user_id"], xp_amount)
    await db.conversations.update_one({"conv_id": body.conversation_id}, {"$set": {"ended": True}})
    return {
        "summary": data.get("summary", ""),
        "strengths": data.get("strengths", ""),
        "improvements": data.get("improvements", ""),
        "words": data.get("words", []),
        "words_saved": saved,
        "xp_earned": xp_amount,
        "streak": prog["streak"],
    }


@api_router.post("/conversation/transcribe")
async def conversation_transcribe(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    suffix = Path(file.filename or "audio.m4a").suffix or ".m4a"
    if suffix.lstrip(".").lower() not in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
        suffix = ".m4a"
    content = await file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as fh:
            result = await oai.audio.transcriptions.create(
                file=fh, model="whisper-1", response_format="json", language="nl"
            )
        transcript = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else str(result))
    except Exception as e:
        logger.error(f"Conversation STT error: {e}")
        raise HTTPException(status_code=502, detail="Transcription indisponible")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return {"text": transcript or ""}


@api_router.get("/")
async def root():
    return {"message": "Tulipe API"}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await db.users.create_index("email", unique=True)
    await db.users.create_index("user_id", unique=True)
    await db.user_sessions.create_index("session_token", unique=True)
    await db.user_sessions.create_index("user_id")
    await db.flashcards.create_index("user_id")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
