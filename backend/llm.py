"""אינטראקציה עם OpenAI לתשובות צ'אט מבוססות מקורות."""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from rag import SearchResult

CHAT_MODEL = "gpt-4o-mini"  # זול ומספיק טוב לרוב. אפשר להחליף ל-gpt-4o לאיכות גבוהה יותר.

SYSTEM_PROMPT = """\
אתה עוזר וירטואלי של מוסד חינוכי-טיפולי. תפקידך לענות על שאלות של עובדי המוסד \
על סמך מסמכי המוסד בלבד.

חוקים נוקשים:
1. ענה אך ורק על סמך המידע שמופיע בקטעי המקור (CONTEXT) שתקבל. אסור להמציא מידע.
2. אם התשובה לא נמצאת במקורות, אמור במפורש: "אין לי מידע על כך במאגר המסמכים. אנא פנה לגורם המוסמך."
3. בכל תשובה, ציין במפורש מאיזה מסמך לקחת את המידע. השתמש בפורמט: [מקור: שם_המסמך]
4. ענה בעברית, בלשון פשוטה ומכבדת.
5. אם המשתמש שואל על מקרה רגיש (אבחנה רפואית/פסיכיאטרית, החלטה טיפולית קריטית), הוסף בסוף: \
"בנושאים רגישים מומלץ להתייעץ עם מנהל המחלקה או גורם מקצועי במוסד."
6. אל תשתף מידע על אנשים פרטיים (תלמידים, מטופלים) ששמם מופיע במקורות, אלא אם השואל ביקש מידע כללי על תהליך/נוהל.
"""


@dataclass
class ChatAnswer:
    answer: str
    sources: list[str]


def _format_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"--- מקור {i}: {r.source} ---\n{r.text}")
    return "\n\n".join(parts)


def answer_question(
    client: OpenAI,
    question: str,
    results: list[SearchResult],
) -> ChatAnswer:
    if not results:
        return ChatAnswer(
            answer="אין לי עדיין מסמכים במאגר. אנא בקש/י מהמנהל להעלות מסמכים לתיקיית הידע.",
            sources=[],
        )

    context = _format_context(results)
    user_message = (
        f"שאלת המשתמש:\n{question}\n\n"
        f"CONTEXT — קטעי מקור מהמסמכים:\n{context}\n\n"
        f"ענה אך ורק על סמך ה-CONTEXT הזה. ציין מקורות בפורמט [מקור: שם_הקובץ]."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    answer_text = resp.choices[0].message.content or ""
    sources = sorted({r.source for r in results})
    return ChatAnswer(answer=answer_text, sources=sources)
