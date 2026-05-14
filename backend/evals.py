"""Eval framework — בדיקת איכות הבוט מול שאלות-זהב.

שימוש:
    python backend/evals.py

מה זה עושה:
- מריץ רשימה של שאלות מוגדרות מראש מול ה-RAG pipeline
- מציג עבור כל שאלה: השאלה, האם נמצאו מקורות צפויים, ציטוט מ-LLM
- מסכם: כמה אחוז מהשאלות מצאו מקור רלוונטי (recall@k)

זה לא מחליף הערכה אנושית, אבל נותן רגרסיה אוטומטית — אם אעשה שינוי שמשבר משהו, נראה את זה.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from llm import ChatSession, answer_question
from rag import HybridIndex

PROJECT_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent / "data" / "knowledge"
CACHE_PATH = Path(__file__).parent / "data" / ".embed_cache.pkl"


@dataclass
class EvalQuestion:
    """שאלה ב-eval set.

    expected_substrings_in_sources: מילים/ביטויים שצפויים להופיע בלפחות אחד
    מהמקורות שאוחזרו. אם כולם לא נמצאו — הבדיקה נכשלת.

    expected_phrases_in_answer: ביטויים שצפויים להופיע בתשובה. שימוש בזהירות —
    LLM יכול לנסח דברים אחרת. נשתמש בזה לבדיקות חזקות בלבד.

    should_say_no_info: אם True — מצופה שהבוט יגיד שאין לו מידע (שאלה מחוץ לתחום).
    """
    question: str
    expected_substrings_in_sources: list[str]
    expected_phrases_in_answer: list[str] = None
    should_say_no_info: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.expected_phrases_in_answer is None:
            self.expected_phrases_in_answer = []


# שאלות-זהב — בנויות לפי התוכן שיש במאגר (פדגוגיה טיפולית, תוכנית אור, מקרי בוחן)
EVAL_SET: list[EvalQuestion] = [
    EvalQuestion(
        question="מה זה תוכנית אור?",
        expected_substrings_in_sources=["אור"],
        notes="שאלת כיסוי בסיסית — מאגר רחב מכיל מסמכים רבים על תוכנית אור",
    ),
    EvalQuestion(
        question="מהם עקרונות הפדגוגיה הטיפולית?",
        expected_substrings_in_sources=["פדגוגיה", "טיפולית"],
        notes="ליבת המאגר — מצופים מקורות מחוברות הפדגוגיה",
    ),
    EvalQuestion(
        question="מהם הקריטריונים לתכנית הישובים 2026?",
        expected_substrings_in_sources=["קריטריונים", "ישובים"],
        notes="בודק שמספרים ומונחים ספציפיים נמצאים — חוזק של BM25",
    ),
    EvalQuestion(
        question="ספר על מקרה בוחן ממסגרות ייחודיות",
        expected_substrings_in_sources=["מסגרות", "ייחודיות"],
        notes="מצופה למצוא את מסמך מקרה בוחן מסגרות ייחודיות",
    ),
    EvalQuestion(
        question="מה זה קב\"ס?",
        expected_substrings_in_sources=["קבס", "קב"],
        notes="בודק ראשי תיבות עם גרשיים — חשוב לעברית",
    ),
    EvalQuestion(
        question="איך להתמודד עם תלמיד בסיכון בבית ספר יסודי?",
        expected_substrings_in_sources=["סיכון", "יסודי"],
        notes="שאלה מורכבת — מצופה שיחזיר מסמכי כנפיים/אור",
    ),
    EvalQuestion(
        question="מה ההבדל בין פיצה למרגריטה?",
        expected_substrings_in_sources=[],
        should_say_no_info=True,
        notes="שאלה לא רלוונטית — מצופה שיגיד שאין מידע",
    ),
    EvalQuestion(
        question="מי ניצח את גביע העולם 2022?",
        expected_substrings_in_sources=[],
        should_say_no_info=True,
        notes="שאלה כללית מחוץ לתחום — מצופה שיגיד שאין מידע",
    ),
    EvalQuestion(
        question="מה זה תוכנית מל\"א?",
        expected_substrings_in_sources=["מלא", "מל"],
        notes="ראשי תיבות נוספים",
    ),
    EvalQuestion(
        question="איך כותבים תיעוד אירוע דגל?",
        expected_substrings_in_sources=["דגל", "תיעוד", "אירוע"],
        notes="מצופה למצוא את מסמך רשימת התיוג",
    ),
]


def _check_sources(results, expected: list[str]) -> tuple[bool, list[str]]:
    """בודק אם כל הביטויים הצפויים מופיעים באחד המקורות שאוחזרו.

    Returns:
        (passed, missing_phrases)
    """
    if not expected:
        return True, []
    all_text = " ".join(f"{r.source} {r.section} {r.text}" for r in results)
    missing = [p for p in expected if p not in all_text]
    return len(missing) == 0, missing


def _check_no_info_response(answer: str) -> bool:
    """בודק אם התשובה מציינת שאין מידע."""
    indicators = ["לא מופיע", "אין לי מידע", "לא נמצא", "מומלץ לפנות"]
    return any(ind in answer for ind in indicators)


def run_eval():
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY לא מוגדר ב-.env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print("🔧 בונה אינדקס...")
    index = HybridIndex(client=client, cache_path=CACHE_PATH)
    summary = index.build(KNOWLEDGE_DIR)
    print(f"   {summary['documents']} מסמכים, {summary['chunks']} chunks")
    print()

    total = len(EVAL_SET)
    passed_retrieval = 0
    passed_answer = 0
    print("=" * 80)
    print(f" Eval: {total} שאלות")
    print("=" * 80)
    print()

    for i, eq in enumerate(EVAL_SET, 1):
        print(f"[{i}/{total}] ❔ {eq.question}")
        if eq.notes:
            print(f"     הערה: {eq.notes}")

        results = index.search(eq.question, k=6)
        # בדיקת retrieval
        if eq.should_say_no_info:
            retrieval_ok = True  # לא משנה מה אוחזר
        else:
            retrieval_ok, missing = _check_sources(results, eq.expected_substrings_in_sources)
            if retrieval_ok:
                passed_retrieval += 1
                print(f"     ✅ Retrieval: נמצא {len(results)} מקורות, כל הביטויים נמצאו")
            else:
                print(f"     ❌ Retrieval: חסרים בביטויים מהמקורות: {missing}")

        if eq.should_say_no_info:
            # רץ דרך LLM ובודק שאומר שאין מידע
            session = ChatSession()
            answer = answer_question(client, eq.question, results, session)
            short = answer[:150].replace("\n", " ")
            if _check_no_info_response(answer):
                passed_answer += 1
                print(f"     ✅ Answer (no-info): \"{short}...\"")
            else:
                print(f"     ❌ Answer (no-info): הבוט לא אמר שאין מידע. תשובה: \"{short}...\"")
        else:
            # רץ דרך LLM ובודק שיש ציטוט מקור
            session = ChatSession()
            answer = answer_question(client, eq.question, results, session)
            short = answer[:200].replace("\n", " ")
            if "[מקור:" in answer or "מקור:" in answer:
                passed_answer += 1
                print(f"     ✅ Answer: כולל ציטוט. \"{short}...\"")
            else:
                print(f"     ⚠️ Answer: חסר ציטוט מסודר. \"{short}...\"")

        # מציג את 3 המקורות העליונים
        for j, r in enumerate(results[:3], 1):
            section = f" § {r.section}" if r.section else ""
            ranks = []
            if r.vector_rank: ranks.append(f"vec#{r.vector_rank}")
            if r.bm25_rank: ranks.append(f"bm25#{r.bm25_rank}")
            print(f"        {j}. [{', '.join(ranks)}] {r.source}{section}")
        print()

    print("=" * 80)
    print(f" סיכום: ")
    print(f"   Retrieval: {passed_retrieval}/{total} ({100*passed_retrieval//total}%)")
    print(f"   Answer:    {passed_answer}/{total} ({100*passed_answer//total}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_eval()
