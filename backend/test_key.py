"""בדיקה מהירה של מפתח OpenAI — בלי טעינת מסמכים."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env")

key = os.environ.get("OPENAI_API_KEY", "")

print("=" * 60)
print(" בדיקת מפתח OpenAI")
print("=" * 60)

if not key:
    print("❌ הקובץ .env לא קיים, או שאין בו OPENAI_API_KEY")
    sys.exit(1)

# מידע על המפתח שנקרא
print(f"אורך המפתח שנקרא:       {len(key)} תווים")
print(f"מתחיל ב:                {key[:10]!r}")
print(f"מסתיים ב:               {key[-6:]!r}")
print(f"מכיל רווחים?            {'כן' if ' ' in key else 'לא'}")
print(f"מכיל ירידת שורה?         {'כן' if chr(10) in key or chr(13) in key else 'לא'}")
print(f"מכיל גרשיים?            {chr(34) in key or chr(39) in key}")
print()

# בדיקה מול OpenAI
print("מנסה להתחבר ל-OpenAI...")
try:
    client = OpenAI(api_key=key)
    # קריאה קטנה ביותר — embedding של מילה אחת
    resp = client.embeddings.create(model="text-embedding-3-small", input="שלום")
    print(f"✅ המפתח עובד! קיבלתי vector באורך {len(resp.data[0].embedding)}")
    print()
    print("הכל בסדר. אפשר להריץ את הבוט.")
except AuthenticationError as e:
    print(f"❌ שגיאת אימות (401):")
    print(f"   {e.message if hasattr(e, 'message') else e}")
    print()
    print("מה שכנראה הסיבה:")
    print("  1. המפתח לא תקין — תיצור חדש ב-https://platform.openai.com/api-keys")
    print("  2. אין לך credit / billing — בדוק ב-https://platform.openai.com/settings/organization/billing")
    print("  3. המפתח שייך לפרויקט שאין לו גישה לדגמים האלה.")
except Exception as e:
    print(f"❌ שגיאה אחרת: {type(e).__name__}: {e}")
