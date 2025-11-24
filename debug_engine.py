from app.core.engine import IndxAI_OS


def test_extraction():
    os = IndxAI_OS()
    query = "my last gmail from render.com"
    clean_q = os._extract_keywords(query)
    print(f"Original: '{query}'")
    print(f"Cleaned: '{clean_q}'")

    # Also test if it actually fetches emails
    print("Fetching emails with cleaned query...")
    emails = os.gmail.get_relevant_emails(clean_q)
    print(f"Found {len(emails)} emails.")
    if emails and "System:" in emails[0]:
        print(f"Error/Message: {emails[0]}")
    else:
        print(f"First email: {emails[0][:100]}...")


if __name__ == "__main__":
    test_extraction()
