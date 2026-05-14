"""LLM wrapper — gpt-4o עם conversation memory, streaming, וציטוט מקורות."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Literal

from openai import OpenAI

from rag import SearchResult

CHAT_MODEL = "gpt-4o"  # איכות גבוהה, תומך בעברית מצוין
MAX_HISTORY_TURNS = 6  # שומר עד 6 turns אחרונות (3 user + 3 assistant)

SYSTEM_PROMPT = """\
את/ה עוזר/ת וירטואלי/ת של אגף בכיר חינוך ילדים ונוער בסיכון, משרד החינוך.
תפקידך לענות על שאלות של אנשי המקצוע באגף — מפקחים, מנהלי תוכניות, מורים, יועצים ומטפלים.

תפעל/י לפי החוקים הבאים בכל תשובה:

1. **התבסס/י רק על המקורות**: ענה/י אך ורק על סמך הקטעים שמופיעים תחת CONTEXT.
   אל תוסיף/י ידע כללי, אל תמציא/י, ואל תשתמש/י במידע שאת/ה "יודע/ת" מחוץ למקורות.

2. **אם אין תשובה במקורות**, אמור/י במפורש:
   "המידע הזה לא מופיע במאגר המסמכים של האגף. מומלץ לפנות ל[רכז התוכנית הרלוונטית / המפקח/ת הממונה / יחידת התוכן]."

3. **ציטוט מקורות בכל תשובה**: בסוף כל טענה משמעותית, ציין/י את המקור בפורמט:
   [מקור: שם_הקובץ § שם_הסעיף] — אם יש סעיף, או [מקור: שם_הקובץ] אם אין.

4. **תשובה מובְנית**:
   - פתח/י במשפט סיכום של 1-2 שורות.
   - אם רלוונטי, פרט/י בנקודות.
   - אם השאלה דורשת פעולה — תן/י סדר פעולות ברור.

5. **שפה**: עברית רהוטה ומקצועית, פניה ניטרלית מבחינה מגדרית כשאפשר.

6. **רגישות**: בנושאי אבחון, החלטות טיפוליות, מקרים פרטניים — סיים/י עם:
   "החלטה מקצועית בנושא זה דורשת התייעצות עם הגורם הממונה והכרה של פרטי המקרה."

7. **שאלות המשך**: אם המשתמש שואל שאלת המשך, התייחס/י להקשר השיחה הקודמת
   אבל בדוק/י שוב את המקורות לתשובה החדשה.

אם המקורות סותרים זה את זה — ציין/י זאת במפורש ותן/י את שני הצדדים עם המקורות.
"""


@dataclass
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ChatSession:
    history: list[ChatTurn] = field(default_factory=list)

    def add(self, role: Literal["user", "assistant"], content: str) -> None:
        self.history.append(ChatTurn(role=role, content=content))
        # שמור על מקסימום turns אחרונות
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history = self.history[-MAX_HISTORY_TURNS:]

    def to_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.history]


def _format_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        section_info = f" § {r.section}" if r.section else ""
        parts.append(
            f"--- מקור {i}: {r.source}{section_info} ---\n{r.text}"
        )
    return "\n\n".join(parts)


def _build_user_message(question: str, results: list[SearchResult]) -> str:
    if not results:
        return (
            f"שאלת המשתמש: {question}\n\n"
            f"לא נמצאו מקורות במאגר. ענה/י לפי כלל מס' 2."
        )
    context = _format_context(results)
    return (
        f"שאלת המשתמש:\n{question}\n\n"
        f"CONTEXT — קטעי מקור:\n{context}\n\n"
        f"ענה/י על השאלה לפי החוקים. ציטוטים בפורמט [מקור: שם_הקובץ § סעיף]."
    )


def stream_answer(
    client: OpenAI,
    question: str,
    results: list[SearchResult],
    session: ChatSession,
) -> Generator[str, None, str]:
    """גנרטור — מחזיר chunks של התשובה תוך כדי כתיבה. מחזיר את התשובה המלאה בסוף."""
    user_message = _build_user_message(question, results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session.to_messages())
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_answer += delta
            yield delta
    return full_answer


def answer_question(
    client: OpenAI,
    question: str,
    results: list[SearchResult],
    session: ChatSession,
) -> str:
    """גרסה לא-streaming — מחזיר את התשובה המלאה כמחרוזת."""
    parts: list[str] = []
    gen = stream_answer(client, question, results, session)
    try:
        while True:
            parts.append(next(gen))
    except StopIteration as stop:
        return "".join(parts) or (stop.value or "")
