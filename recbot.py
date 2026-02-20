# Final Project

import discord
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
from discord.ext import commands
from surprise import Dataset, Reader, SVD
load_dotenv()
GUILD = 'Input_Your_Server_Name_Here'
intents = discord.Intents.default()
intents.message_content = True

# API Key to use
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

system_prompt = {"role": "system", "content": "you are a helpful assistant"}

user_messages = {}

# command prefix is added to the start of each command name to create the command
# this bot will listen for !add_user !search !rate and !rec
bot = commands.Bot(command_prefix='!', intents=intents)

# dictionary maps the discord user id to their movielens user_id (in the u.user file)
user_id_mapping= {}
# dictionary maps the discord username to their movielens user_id
username_mapping = {}

# dictionary maps the movie title to the movielens movie_id (in the u.item file)
movie_titles = {}

# next movielens user_id we should assign to a discord user
# after assigning this id, we need to increment it
# that way each new discord user will have a unique movielens user_id
next_user_id = 0

# filepaths to the user, item, and data files
user_filepath = './ml-100k/u.user'
data_filepath = './ml-100k/u.data'
movie_filepath = './ml-100k/u.item'


# called when the bot is started
# load in all registered discord users
def load_users():
    global next_user_id
    # the last two columns in the table are being co-opted to store the username and discord id
    # they normally contain the occupation and zip code, but we don't use this information in our model
    # it is free realestate!
    user_data = pd.read_csv(user_filepath, delimiter="|", names=['user_id','age', 'gender','discord_username','discord_user_id'])

    # for each row in the dataset
    for _, row in user_data.iterrows():
        # discord users in the dataset have their gender marked as D
        if row['gender'] != 'D':
            continue
        # add the user to the dictionaries
        user_id_mapping[str(row['discord_user_id'])] = row['user_id']
        username_mapping[row['discord_username']] = row['user_id']
    # find the largest user_id that already exists in the table and add 1
    # this will be assigned to the next discord user we register
    next_user_id = max(user_data['user_id'].tolist(), default=0) + 1

# called when the bot is started
# creates a dictionary that maps the movie title the movie_id for every movie in the table
def load_movies(file='./ml-100k/u.item'):
    # movie file is not utf-8 encoded, we have to provide the correct encoding to avoid an error
    item_data = pd.read_csv(file, delimiter='|', encoding='ISO-8859-1', usecols=[0, 1], names=['movie_id', 'title'])
    for _, row in item_data.iterrows():
        movie_titles[row['title']] = row['movie_id']


# initialize our dictionaries
load_users()
load_movies()

# decorators change the functionality of functions
# this one marks it as a handler for the ready event
# this function will trigger exactly once, when the bot starts up
@bot.event
async def on_ready():
    print(f'{bot.user} is now running!')

# Add user command
# this decorator marks the function as a handler for the !add_user command
# we can name the function anything we want, but we use add_user to be consistent
@bot.command(name='add_user', help = 'Registers the user to the system. Usage: !add_user')
async def add_user(ctx):
    # we modify next_user_id in this function, it has to be declared global or else we end up shadowing it instead
    global next_user_id
    discord_user_id = str(ctx.author.id)
    discord_user_name = ctx.author.name
    # first check to see if the user is already registered
    if discord_user_name in username_mapping:
        # we don't have to do anything, notify the user
        await ctx.send("you are already registered!")
        return
    # notice that we do not use an else block but instead return early
    # this makes our code much easier to read!

    # since we got here, user is not already registered: we register them
    username_mapping[discord_user_name] = next_user_id
    user_id_mapping[discord_user_id] = next_user_id
    # we have to update the next_user_id
    next_user_id += 1
    # how do we remember users we have added after turning the program off?
    # add the row to the file!
    new_user_data = f"{next_user_id}|18|D|{discord_user_name}|{discord_user_id}\n"
    # 'a' means that we open the file in append mode, we just write new stuff at the end of the file
    # write does not automatically add new lines, we have to make sure we add a trailing new line so that the structure
    # of the file is maintained. When in doubt: just run the code and check on the file yourself
    with open('./ml-100k/u.user', 'a') as f:
        f.write(new_user_data)

    # notify the user that they have been registered
    await ctx.send(f"Hi {discord_user_name}, you have id {discord_user_id}, internal id {next_user_id}")

# since users don't know what movie id goes with which movie, we need a function that will give them that information
# this command takes a portion of a movie title and finds all movies in the list containing the title fragment
# the code lists the movie_id then the full title

# Search command
# since title_search appears after * in the parameter list, that means that all words after !search are included in the variable
# any variable before * will only hold one word at a time
@bot.command(name='search', help = 'Searches for a movie by title. Usage: !search <movie_title>')
async def search(ctx,*, title_search: str):
    # place to keep matches
    matches = []
    # iterate over the keys and values of movie_titles
    for title, id in movie_titles.items():
        # check if title search is contained in the title
        if title_search.lower() in title.lower():
            # it is, add it to the list
            matches.append((title, id))
    result = ""

    total_matches = len(matches)  # Total number of matches found

    # exit early if no matches were found
    # this is more likely than not because the database is very limited
    if len(matches) == 0:
        await ctx.send("Could not find any movies that match your search!")
        return

    # Truncate results if too many matches
    truncated = False
    # limit results in case they search for "the"
    if len(matches) > 10:
        matches = matches[0:10]
        truncated = True

    # format the results in a nice string
    message = "Here are the search results:\n"
    for item in matches:
        message += f"{item[1]}: {item[0]}\n"

    # Add a warning if the results were truncated
    if truncated:
        message += "\n⚠️ Too many results! Showing the first 10 matches out of "
        message += f"{total_matches} total. Please refine your search."

    # post the results for the user to see
    await ctx.send(message)

# Rating Command
# add the users rating for the movie to the u.data table
# expected format: !rate movie_id rating
# if the user mixes up the movie_id and the rating then they are going to get bad recommendations
# this is why we want to use a more robust interface!
@bot.command(name='rate', help = 'Rates a movie. Usage: !rate <movie_id> <rating> (rating must be between 1 and 5)')
async def rate(ctx, movie_id: int, rating: int):
    # we cannot add users if they do not have a user_id already
    if ctx.author.name not in username_mapping:
        await ctx.send("Use !add_user to register yourself before rating movies.")
        return

    user_id = username_mapping[ctx.author.name]
    # Ensure the rating is within [1,5]
    if rating < 1:
        rating = 1
    if rating > 5:
        rating = 5
    # Load existing ratings
    existing_data = pd.read_csv(data_filepath, delimiter='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Check if the user already rated this movie
    existing_rating = existing_data[
        (existing_data['user_id'] == user_id) & (existing_data['movie_id'] == movie_id)
        ]
    if not existing_rating.empty:
        await ctx.send("You have already rated this movie. Update your rating instead!")
        return

    # Add new rating to the file
    with open(data_filepath, 'a') as f:
        f.write(f"{user_id}\t{movie_id}\t{rating}\t0\n")
    # Notify the user
    await ctx.send("Your rating has been registered. Thank you!")

# Update Rating command
# update the users movie rating of a movie they have already rated.
# expected format: !update_rating movie_id rating
@bot.command(name='update_rating', help='Updates your existing rating for a movie. Usage: !update_rating <movie_id> <new_rating>')
async def update_rating(ctx, movie_id: int, new_rating: int):
    # Ensure the user is registered
    if ctx.author.name not in username_mapping:
        await ctx.send("Use !add_user to register yourself before updating ratings.")
        return

    user_id = username_mapping[ctx.author.name]
    # Ensure the new rating is within [1,5]
    if new_rating < 1:
        new_rating = 1
    if new_rating > 5:
        new_rating = 5
    # Load existing ratings
    existing_data = pd.read_csv(data_filepath, delimiter='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Check if the user has already rated this movie
    index_to_update = existing_data[
        (existing_data['user_id'] == user_id) & (existing_data['movie_id'] == movie_id)
        ].index
    if index_to_update.empty:
        await ctx.send("You have not rated this movie yet. Use !rate to add a new rating.")
        return

    # Update the rating in the DataFrame
    existing_data.loc[index_to_update, 'rating'] = new_rating
    # Save the updated data back to the file
    existing_data.to_csv(data_filepath, sep='\t', header=False, index=False)
    # Notify the user
    await ctx.send("Your rating has been updated. Thank you!")

# Summary Command
@bot.command(name='summary', help='Lists all the rating you have submitted. Usage: !summary')
async def summary (ctx):
    # Ensure the user is registered
    if ctx.author.name not in username_mapping:
        await ctx.send("Use !add_user to register yourself before viewing your summary.")
        return

    user_id = username_mapping[ctx.author.name]

    # Load existing ratings
    existing_data = pd.read_csv(data_filepath, delimiter='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    user_ratings = existing_data[existing_data['user_id'] == user_id]

    # If no ratings are found
    if user_ratings.empty:
        await ctx.send("You have not submitted any ratings yet.")
        return

    if user_ratings.empty:
        await ctx.send("You haven't submitted any ratings yet. Use !rate to add some!")
        return

    # Map movie IDs to titles
    item_data = pd.read_csv(movie_filepath, delimiter='|', encoding='ISO-8859-1',
                            usecols=[0, 1], names=['movie_id', 'title'])
    movie_dict = dict(zip(item_data['movie_id'], item_data['title']))

    # Prepare the summary message
    summary_message = "Here are the ratings you've submitted:\n"
    for _, row in user_ratings.iterrows():
        movie_title = movie_dict.get(row['movie_id'], "Unknown Movie")
        summary_message += f"{movie_title} (ID: {row['movie_id']}): {row['rating']}/5\n"

    # Send the summary to the user
    await ctx.send(summary_message)

# we want a command to predict a rating for a movie
# expected format: !rec movie_id
# user needs to use !search to figure out the movie id from the title
@bot.command(name='rec', help='Predicts a rating for a movie. Usage: !rec <movie_id>')
async def rec(ctx, movie_id: str):
    # cannot make predictions before registering the user
    if ctx.author.name not in username_mapping:
        await ctx.send("use !add_user to register yourself before asking for recommendations")
        return

    # Ensure users have enough ratings before recommendations
    user_id = username_mapping[ctx.author.name]

    # Load existing ratings
    existing_data = pd.read_csv(data_filepath, delimiter='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Check if the user has rated enough movies
    user_ratings_count = existing_data[existing_data['user_id'] == user_id].shape[0]
    if user_ratings_count < 5:
        await ctx.send("You need to rate at least 5 movies before we can give you recommendations.")
        return

    # this code as written actually reads the userIDs and itemIDs as strings instead of integers
    reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
    dataset = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

    # formats the dataset so that it is ready for training the surprise SVD model
    trainset = dataset.build_full_trainset()

    # n_factors determines the number of hidden features/preferences for each user and item
    #     if a user has less reviews than n_factors, the accuracy of the predictions drop precipitously
    # n_epochs number of iterations of the algorithm, larger values makes results better but also takes more time
    # biased penalizes the individual features and preferences for being too large
    #     slightly reduces prediction accuracy
    #     reduces volatility for users or movies that have fewer reviews
    algorithm = SVD(n_factors = 10, n_epochs=200, biased=True)
    algorithm.fit(trainset)

    # call predict using string version of the user_id and string version of the movie_id

    user_id = username_mapping[ctx.author.name]
    rating = algorithm.predict(str(user_id), str(movie_id))

    # check to make sure that the model did not give up
    if rating.details['was_impossible']:
        await ctx.send("model could not produce a rating")
        return

    # notify the user what the rating is
    await ctx.send(f"predicted rating is {rating.est}")

@bot.command(name='synopsis', help='Generates a movie summary using GPT-3.5. Usage: !synopsis <movie_title>')
async def synopsis(ctx, *, movie_title: str):
    try:
        # Send the request to the Open AI for a synopsis
        completion = client.chat.completions.create(
            model = "gpt-3.5-turbo-0125",
            messages = [
                {"role": "system", "content": "You aare a helpful assistant the generates movie synopses."},
                {"role": "user", "content": f"Please provide a summary of the movie '{movie_title}'."}
            ],
            temperature = 0.7, # creativity
            timeout = 30, # waiting time
            max_tokens = 400 # max 400 tokens or about 1600 characters
        )

        print(completion)

        summary = completion.choices[0].message.content.strip()
        await ctx.send(f"**{movie_title}**: {summary}")

    except Exception as e:
        # Handle any exceptions and notify the user
        await ctx.send(f"An error occurred: {str(e)}")

bot.run(os.getenv("TOKEN"))
bot.run(os.getenv("DISCORD_BOT_TOKEN"))
