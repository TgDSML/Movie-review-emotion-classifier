"""
Real-World Application: Manual Mood Analyzer
Simulates analyzing a review provided by the user manually.
"""
from scripts_xgb.infer_xgb import predict_xgboost

def analyze_text_vibe(user_text):
    # 1. Πρόβλεψη
    reviews = [user_text]
    results = predict_xgboost(reviews)
    primary_emotion = results[0]['prediction']
    
    print(f"\n📊 Detected Emotion: {primary_emotion.upper()}")

    # 2. Logic Layer (Heuristic Mapping)
    final_verdict = "Unknown"
    text_lower = user_text.lower()

    # --- ΟΡΙΣΜΟΣ ΛΕΞΕΩΝ ΚΛΕΙΔΙΩΝ ---
    # Λέξεις που δείχνουν ότι η ταινία είναι "σκουπίδι"
    trash_keywords = ["waste", "worst", "boring", "terrible", "money", "trash", "garbage", "stupid", "bad", "awful", "sleep"]
    
    # Λέξεις για δράση
    action_keywords = ["villain", "revenge", "justice", "intense", "fight", "brutal", "grit", "violent", "killer", "action", "blood"]

    # Υπολογισμός σκορ
    trash_score = sum(text_lower.count(w) for w in trash_keywords)
    action_score = sum(text_lower.count(w) for w in action_keywords)

    # --- LOGIC BLOCKS ---

    # ΠΕΡΙΠΤΩΣΗ 1: ANGER (Θυμός)
    if primary_emotion == "anger":
        if action_score > trash_score:
            final_verdict = "👊 High-Octane Action / Revenge Thriller"
        else:
            final_verdict = "🗑️ Trash / Poorly Received (Viewer is Mad)"

    # ΠΕΡΙΠΤΩΣΗ 2: SADNESS (Εδώ κάναμε τη διόρθωση!)
    # Αν βγει Sadness, τσεκάρουμε μήπως ο χρήστης απλά κράζει την ταινία
    elif primary_emotion == "sadness":
        if trash_score > 0:
            # Αν έχει λέξεις όπως "boring", "waste", τότε δεν είναι δράμα, είναι χάλια ταινία!
            final_verdict = "🗑️ Trash / Disappointing Movie (Viewer is Sad/Bored)"
        else:
            final_verdict = "😭 Tearjerker / Drama"

    # ΠΕΡΙΠΤΩΣΗ 3: JOY (Χαρά)
    elif primary_emotion == "joy":
        comedy_keywords = ["funny", "laugh", "hilarious", "joke", "comedy", "humor"]
        romance_keywords = ["love", "beautiful", "couple", "heart", "romantic", "kiss"]
        
        comedy_score = sum(text_lower.count(w) for w in comedy_keywords)
        romance_score = sum(text_lower.count(w) for w in romance_keywords)

        if romance_score > comedy_score:
            final_verdict = "❤️ Romance / Date Movie"
        else:
            final_verdict = "🤣 Comedy / Feel-Good"

    # ΠΕΡΙΠΤΩΣΗ 4: FEAR (Φόβος)
    elif primary_emotion == "fear":
        final_verdict = "👻 Horror / Thriller"

    # ΠΕΡΙΠΤΩΣΗ 5: SURPRISE (Έκπληξη)
    elif primary_emotion == "surprise":
        final_verdict = "🤯 Sci-Fi / Plot Twist / Mystery"

    # ΠΕΡΙΠΤΩΣΗ 6: LOVE (Αγάπη)
    elif primary_emotion == "love":
        final_verdict = "❤️ Romance / Appreciation"

    print("-" * 50)
    print(f"📝 TEXT: \"{user_text[:60]}...\"")
    print(f"🧠 AI VIBE CHECK: {final_verdict}")
    print("-" * 50)

if __name__ == "__main__":
    print("="*60)
    print("🍿 MOVIE MOOD ANALYZER (Corrected Logic)")
    print("Type a review to test.")
    print("="*60)

    while True:
        try:
            user_input = input("\n✍️  Write a review: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input:
                continue
            analyze_text_vibe(user_input)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")