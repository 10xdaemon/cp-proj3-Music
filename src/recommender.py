"""
Music recommendation engine.

Public API:
    Song         — audio feature data class
    UserProfile  — user taste preference data class
    load_songs   — read a CSV catalog into a list of Song objects
    Recommender  — score and rank songs against a UserProfile
"""

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its audio feature attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    speechiness: float = 0.0
    instrumentalness: float = 0.0


@dataclass
class UserProfile:
    """Represents a user's taste preferences used to score songs."""
    preferred_mood: str
    preferred_genre: str
    target_energy: float
    target_tempo_bpm: float
    target_acousticness: float
    target_speechiness: float
    sigma: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Adjacency graph used for mood distance scoring.
# Edges are undirected: if B is in MOOD_GRAPH[A], then A is in MOOD_GRAPH[B].
MOOD_GRAPH: Dict[str, set] = {
    'euphoric':    {'happy'},
    'happy':       {'euphoric', 'uplifted', 'relaxed'},
    'uplifted':    {'happy', 'groovy'},
    'groovy':      {'uplifted', 'bittersweet'},
    'relaxed':     {'happy', 'romantic', 'chill'},
    'romantic':    {'relaxed', 'bittersweet'},
    'bittersweet': {'groovy', 'romantic'},
    'chill':       {'relaxed', 'nostalgic', 'focused'},
    'nostalgic':   {'chill', 'peaceful'},
    'peaceful':    {'nostalgic'},
    'focused':     {'chill', 'moody', 'intense'},
    'moody':       {'focused', 'melancholic', 'dark'},
    'melancholic': {'moody'},
    'intense':     {'focused', 'dark'},
    'dark':        {'moody', 'intense', 'angry'},
    'angry':       {'dark'},
}

# BPM range used to normalise tempo before Gaussian scoring.
_TEMPO_MIN = 52.0
_TEMPO_MAX = 168.0

# Maximum points awarded per scoring dimension (sum = 8.0).
_W_GENRE        = 2.0
_W_ENERGY       = 2.0
_W_ACOUSTICNESS = 1.5
_W_TEMPO        = 1.0
_W_SPEECHINESS  = 0.5

# Mood graph distance → points awarded.
_MOOD_POINTS: Dict[int, float] = {0: 1.0, 1: 0.5, 2: 0.2}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gaussian(diff: float, sigma: float) -> float:
    return math.exp(-(diff ** 2) / (2 * sigma ** 2))


def _mood_distance(user_mood: str, song_mood: str) -> int:
    if user_mood == song_mood:
        return 0
    neighbors = MOOD_GRAPH.get(user_mood, set())
    if song_mood in neighbors:
        return 1
    if any(song_mood in MOOD_GRAPH.get(n, set()) for n in neighbors):
        return 2
    return 99


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Song]:
    """Read a songs CSV file and return a list of Song objects."""
    songs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            songs.append(Song(
                id=int(clean['id']),
                title=clean['title'],
                artist=clean['artist'],
                genre=clean['genre'],
                mood=clean['mood'],
                energy=float(clean['energy']),
                tempo_bpm=float(clean['tempo_bpm']),
                valence=float(clean['valence']),
                danceability=float(clean['danceability']),
                acousticness=float(clean['acousticness']),
                speechiness=float(clean['speechiness']),
                instrumentalness=float(clean['instrumentalness']),
            ))
    return songs


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class Recommender:
    """Scores and ranks a catalog of songs against a user preference profile."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def score(user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Score one song against user preferences.

        Returns:
            (total_score, reasons) where reasons is a list of labeled score components.
        """
        total = 0.0
        reasons = []

        # Genre match: max +2.0
        if song.genre == user.preferred_genre:
            total += _W_GENRE
            reasons.append(f'genre match (+{_W_GENRE})')

        # Mood adjacency: +1.0 exact / +0.5 one step / +0.2 two steps
        dist = _mood_distance(user.preferred_mood, song.mood)
        mood_pts = _MOOD_POINTS.get(dist, 0.0)
        if mood_pts > 0:
            label = 'match' if dist == 0 else f'{dist}-step away'
            total += mood_pts
            reasons.append(f'mood {label} (+{mood_pts})')

        # Continuous feature proximity via Gaussian (weighted by importance)
        norm_user_tempo = (user.target_tempo_bpm - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)
        norm_song_tempo = (song.tempo_bpm        - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)

        for weight, diff, label in (
            (_W_ENERGY,       abs(user.target_energy       - song.energy),       'energy proximity'),
            (_W_ACOUSTICNESS, abs(user.target_acousticness - song.acousticness), 'acousticness proximity'),
            (_W_TEMPO,        abs(norm_user_tempo - norm_song_tempo),            'tempo proximity'),
            (_W_SPEECHINESS,  abs(user.target_speechiness  - song.speechiness),  'speechiness proximity'),
        ):
            pts = weight * _gaussian(diff, user.sigma)
            total += pts
            reasons.append(f'{label} (+{pts:.2f})')

        return total, reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs ranked by score for the given user profile.

        A genre-streak cap (max 2 consecutive same-genre songs) is applied to
        encourage variety in the final list.
        """
        ranked = sorted(self.songs, key=lambda s: self.score(user, s)[0], reverse=True)

        results: List[Song] = []
        last_genre: str | None = None
        streak = 0

        for song in ranked:
            streak = streak + 1 if song.genre == last_genre else 1
            last_genre = song.genre
            if streak <= 2:
                results.append(song)
            if len(results) == k:
                break

        return results

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        _, reasons = self.score(user, song)
        return ' | '.join(reasons)
