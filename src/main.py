"""
Command line runner for the Music Recommender Simulation.

This file demonstrates the recommender using three preset user profiles.

The implementation lives in recommender.py:
- load_songs         — reads the CSV catalog into Song objects
- Recommender        — scores and ranks songs against a UserProfile
- Recommender.score  — breaks down a song's score into labeled components
"""

from src.recommender import load_songs, Recommender, UserProfile


PROFILES = {
    "High-Energy Pop": UserProfile(
        preferred_mood="happy",
        preferred_genre="pop",
        target_energy=0.88,
        target_tempo_bpm=125,
        target_acousticness=0.10,
        target_speechiness=0.06,
        sigma=0.20,
    ),
    "Chill Lofi": UserProfile(
        preferred_mood="chill",
        preferred_genre="lofi",
        target_energy=0.38,
        target_tempo_bpm=76,
        target_acousticness=0.80,
        target_speechiness=0.02,
        sigma=0.20,
    ),
    "Deep Intense Rock": UserProfile(
        preferred_mood="intense",
        preferred_genre="rock",
        target_energy=0.93,
        target_tempo_bpm=152,
        target_acousticness=0.10,
        target_speechiness=0.06,
        sigma=0.15,
    ),
}


def print_recommendations(label: str, recommendations: list, user: UserProfile) -> None:
    print("\n" + "=" * 50)
    print(f"  {label}  —  Top {len(recommendations)} Recommendations")
    print("=" * 50)
    for i, song in enumerate(recommendations, start=1):
        score, reasons = Recommender.score(user, song)
        print(f"\n#{i}  {song.title}  —  {song.artist}")
        print(f"    Genre: {song.genre}  |  Mood: {song.mood}")
        print(f"    Score: {score:.2f} / 8.0")
        print("    Reasons:")
        for reason in reasons:
            print(f"      • {reason}")
    print("\n" + "=" * 50)


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    rec = Recommender(songs)
    for label, user in PROFILES.items():
        recommendations = rec.recommend(user, k=5)
        print_recommendations(label, recommendations, user)


if __name__ == "__main__":
    main()
