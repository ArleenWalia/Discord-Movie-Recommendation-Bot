# 🎬 CineBot — Discord Movie Recommendation Bot

CineBot is a Discord bot that provides personalized movie recommendations using collaborative filtering (SVD) and AI-generated synopses via the OpenAI API. It uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) as its movie and ratings database.

---

## Requirements

- Python 3.9+
- A Discord Bot Token
- An OpenAI API Key
- The MovieLens 100K dataset (`ml-100k/`)
- The following Python packages:
  ```
  discord.py
  openai
  scikit-surprise
  pandas
  python-dotenv
  ```

Install dependencies:
```bash
pip install discord.py openai scikit-surprise pandas python-dotenv
```

---

## Setup

1. Download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) and place the `ml-100k/` folder in the project root.

2. Create a `.env` file in the project root:
   ```env
   TOKEN=your_discord_bot_token
   OPEN_AI_KEY=your_openai_api_key
   ```

3. In `bot.py`, set your Discord server name:
   ```python
   GUILD = 'Your_Server_Name_Here'
   ```

4. Run the bot:
   ```bash
   python bot.py
   ```

---

## Commands

| Command | Description |
|---|---|
| `!add_user` | Registers you with the bot before you can rate movies or get recommendations |
| `!search <title>` | Searches for a movie by title and returns matching results with their movie IDs |
| `!rate <movie_id> <rating>` | Rates a movie from 1–5. If you've already rated it, you'll be prompted to use `!update_rating` |
| `!update_rating <movie_id> <new_rating>` | Updates a rating you've already submitted |
| `!summary` | Lists all movies you've rated along with your scores |
| `!rec <movie_id>` | Predicts how much you'd enjoy a movie based on your ratings using SVD. Requires at least 5 ratings |
| `!synopsis <movie_title>` | Generates an AI-written synopsis for any movie using the OpenAI API |

> Use `!help <command>` for detailed usage info on any command.

---

## Notes

- **`!synopsis`** makes a fresh, stateless request to the OpenAI API each time — no conversation history is retained — to keep API costs low.
- **`!rec`** uses SVD with 10 latent factors. You'll need at least 5 ratings for the model to produce a prediction. Fewer ratings than latent factors will reduce prediction accuracy.
- Ratings are stored directly in the MovieLens `u.data` file and users are stored in `u.user`, so all data persists across restarts.
