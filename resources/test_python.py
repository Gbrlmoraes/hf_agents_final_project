import random


def get_mood() -> str:
    moods = ["happy", "sad", "angry", "excited", "bored"]
    return f"I'm feeling {random.choice(moods)} today!"


if __name__ == "__main__":
    print(get_mood())
