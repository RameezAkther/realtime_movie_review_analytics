import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from wordcloud import WordCloud
import base64
import io
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
rf_model_loaded = joblib.load('./models/rf_model.joblib')
tfidf_vect_loaded = joblib.load('./models/tfidf_vect.joblib')

url = "https://open-ai21.p.rapidapi.com/conversationllama"

headers = {
	"x-rapidapi-key": "d4f11cda99mshaf0c893d26831e1p1e08ecjsnf73b7a997619",
	"x-rapidapi-host": "open-ai21.p.rapidapi.com",
	"Content-Type": "application/json"
}

def extractive_summary(text, num_sentences=15):
    """
    Function to generate extractive summary without using external summarization libraries.
    This method uses TF-IDF to extract important sentences from the input text.
    
    :param text: Input text to summarize.
    :param num_sentences: Number of sentences to include in the summary.
    :return: Extracted summary as a string.
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Convert sentences into TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # Calculate cosine similarities between all sentences (cosine similarity matrix)
    cosine_similarities = (tfidf_matrix * tfidf_matrix.T).toarray()  # Compute similarity matrix
    
    # Calculate sentence scores by summing the cosine similarities of each sentence with all others
    sentence_scores = cosine_similarities.sum(axis=1)
    
    # Get the indices of the most similar sentences (importance of each sentence)
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-num_sentences:]]
    
    # Join the most important sentences into a summary
    summary = ' '.join(ranked_sentences)
    return summary

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words and apply stemming
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    # Join tokens back into a single string
    processed_text = ' '.join(cleaned_tokens)
    return processed_text

# Vectorizing function
def vectorize(data, tfidf_vect_fit):
    X_tfidf = tfidf_vect_fit.transform(data)
    words = tfidf_vect_fit.get_feature_names_out()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return X_tfidf_df

# Function to predict sentiment on new reviews
def predict_sentiment(new_reviews):
    # Preprocess and transform using loaded vectorizer
    clean_reviews = [preprocess_text(review) for review in new_reviews]
    reviews_tfidf = tfidf_vect_loaded.transform(clean_reviews)
    # Predict sentiment using loaded model
    predictions = rf_model_loaded.predict(reviews_tfidf)
    return predictions

def make_good(text, movie_name,val):
    if val == 1:
        prompt = f"{text}\nThe above text is the positive extractive summary of a movie reviews. Please rewrite the text in a more positive tone and meaningfull. IF there there is no extractive summary text just write a positve review of your own for the movie{movie_name}. Also don't add your sentences like I would like to give you a more positive review, just give the summary please"
    else:
        prompt = f"{text}\nThe above text is the negative extractive summary of a movie reviews. Please rewrite the text in a negative tone and meaningfull. IF there there is no extractive summary text just write a negative review of your own for the movie{movie_name}. Also don't add your sentences like I would like to give you a negative review, just give the summary please"
    payload = {
	"messages": [
		{
			"role": "user",
			"content": prompt
		}
	],
	"web_access": False
    }
    response = requests.post(url, json=payload, headers=headers)
    response = response.json()
    return response['result']

def summarize_reviews(movie_details):
    positive_summary = extractive_summary(" ".join(movie_details['positive_reviews']))
    negative_summary = extractive_summary(" ".join(movie_details['negative_reviews']))
    positive_summary = make_good(positive_summary,movie_details['name'],1)
    negative_summary = make_good(negative_summary,movie_details['name'],0)
    movie_details['positive_review_summary'] = positive_summary
    movie_details['negative_review_summary'] = negative_summary

def generate_wordcloud(reviews):
    # Combine reviews into a single string for the word cloud
    text = " ".join(reviews)
    wordcloud = WordCloud(width=400, height=200, background_color="white").generate(text)
    
    # Save to a BytesIO object and encode as base64
    img_buffer = BytesIO()
    wordcloud.to_image().save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"

def plot_review_sentiment_graph(movie_details):
    reviews = movie_details['reviews']
    # Transform reviews into TF-IDF feature vectors
    reviews_tfidf = tfidf_vect_loaded.transform(reviews)
    
    # Compute cosine similarity between reviews
    similarity_matrix = cosine_similarity(reviews_tfidf)

    # Create a graph
    G = nx.Graph()
    for i, review in enumerate(reviews):
        G.add_node(i, label=f"Review {i+1}")
    
    # Add edges for similar reviews (threshold can be adjusted)
    threshold = 0.8
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()})
    plt.title("Review Sentiment Graph")
    
    # Save the plot to a BytesIO object
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="PNG")
    img_buffer.seek(0)
    plt.close()

    # Encode the image as Base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def plot_word_cooccurrence_graph(reviews):
    from itertools import combinations
    from collections import Counter

    # Preprocess text
    all_words = [word for review in reviews for word in review.lower().split() if word not in stop_words]
    
    # Find word pairs (co-occurrences)
    cooccurrences = Counter()
    for review in reviews:
        words = [word for word in review.lower().split() if word not in stop_words]
        cooccurrences.update(combinations(words, 2))

    # Create a graph
    G = nx.Graph()
    for (word1, word2), count in cooccurrences.items():
        if count > 5:  # Filter weak connections
            G.add_edge(word1, word2, weight=count)

    # Draw the graph
    pos = nx.spring_layout(G)
    node_sizes = [500 + 50 * G.degree(n) for n in G.nodes()]
    edge_widths = [0.1 * G[u][v]['weight'] for u, v in G.edges()]
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_color='lightgreen', 
        node_size=node_sizes, edge_color=edge_widths, edge_cmap=plt.cm.Blues
    )
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Word Co-occurrence Graph")
    
    # Save the plot to a BytesIO object
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="PNG")
    img_buffer.seek(0)
    plt.close()

    # Encode the image as Base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def classify_summarize_reviews(movie_details):
    reviews = movie_details['reviews']
    predictions = predict_sentiment(reviews)
    
    positive_reviews = [reviews[i] for i in range(len(reviews)) if predictions[i] == 1]
    negative_reviews = [reviews[i] for i in range(len(reviews)) if predictions[i] == 0]
    
    movie_details['positive_reviews'] = positive_reviews
    movie_details['negative_reviews'] = negative_reviews
    summarize_reviews(movie_details)
    
    # Generate word clouds
    movie_details['positive_wordcloud'] = generate_wordcloud(positive_reviews)
    movie_details['negative_wordcloud'] = generate_wordcloud(negative_reviews)
    
    # Generate graph visualizations
    movie_details['sentiment_graph'] = plot_review_sentiment_graph(movie_details)
    movie_details['word_cooccurrence_graph'] = plot_word_cooccurrence_graph(reviews)
    
    return movie_details



'''
reviews = ['One of the early scenes in "Pulp Fiction" features two hit-men discussing what a Big Mac is called in other countries. Their dialogue is witty and entertaining, and it\'s also disarming, because it makes these two thugs seem all too normal. If you didn\'t know better, you might assume these were regular guys having chit-chat on their way to work. Other than the comic payoff at the end of the scene, in which they use parts of this conversation to taunt their victims, their talk has no relevance to anything in the film, or to anything else, for that matter. Yet without such scenes, "Pulp Fiction" wouldn\'t be "Pulp Fiction." I get the sense that Tarantino put into the film whatever struck his fancy, and somehow the final product is not only coherent but wonderfully textured.\n\nIt\'s no wonder that fans spend so much time debating what was in the suitcase, reading far more into the story than Tarantino probably intended. The film is so intricately structured, with so many astonishing details, many of which you won\'t pick up on the first viewing, that it seems to cry out for some deeper explanation. But there is no deeper explanation. "Pulp Fiction," is, as the title indicates, purely an exercise in technique and style, albeit a brilliant and layered one. Containing numerous references to other films, it is like a great work of abstract art, or "art about art." It has all the characteristics we associate with great movies: fine writing, first-rate acting, unforgettable characters, and one of the most well-constructed narratives I\'ve ever seen in a film. But to what end? The self-contained story does not seem to have bearing on anything but itself.\n\nThe movie becomes a bit easier to understand once you realize that it\'s essentially a black comedy dressed up as a crime drama. Each of the three main story threads begins with a situation that could easily form the subplot of any standard gangster movie. But something always goes wrong, some small unexpected accident that causes the whole situation to come tumbling down, leading the increasingly desperate characters to absurd measures. Tarantino\'s originality stems from his ability to focus on small details and follow them where they lead, even if they move the story away from conventional plot developments.\n\nPerhaps no screenplay has ever found a better use for digressions. Indeed, the whole film seems to consist of digressions. No character ever says anything in a simple, straightforward manner. Jules could have simply told Yolanda, "Be cool and no one\'s going to get hurt," which is just the type of line you\'d find in a generic, run-of-the-mill action flick. Instead, he goes off on a tangent about what Fonzie is like. Tarantino savors every word of his characters, finding a potential wisecrack in every statement and infusing the dialogue with clever pop culture references. But the lines aren\'t just witty; they are full of intelligent observations about human behavior. Think of Mia\'s statement to Vincent, "That\'s when you know you\'ve found somebody special: when you can just shut the f--- up for a minute and comfortably enjoy the silence."\n\nWhat is the movie\'s purpose exactly? I\'m not sure, but it does deal a lot with the theme of power. Marsellus is the sort of character who looms over the entire film while being invisible most of the time. The whole point of the big date sequence, which happens to be my favorite section of the film, is the power that Marsellus has over his men without even being present. This power is what gets Vincent to act in ways you would not ordinarily expect from a dumb, stoned gangster faced with an attractive woman whose husband has gone away. The power theme also helps explain one of the more controversial aspects of the film, its liberal use of the N-word. In this film, the word isn\'t just used as an epithet to describe blacks: Jules, for instance, at one point applies the term to Vincent. It has more to do with power than with race. The powerful characters utter the word to express their dominance over weaker characters. Most of these gangsters are not racist in practice. Indeed, they are intermingled racially, and have achieved a level of equality that surpasses the habits of many law-abiding citizens in our society. They resort to racial epithets because it\'s a patter that establishes their separateness from the non-criminal world.\n\nThere\'s a nice moral progression to the stories. We presume that Vincent hesitates to sleep with Mia out of fear rather than loyalty. Later, Butch\'s act of heroism could be motivated by honor, but we\'re never sure. The film ends, however, with Jules making a clear moral choice. Thus, the movie seems to be exploring whether violent outlaws can act other than for self-preservation.\n\nStill, it\'s hard to find much of a larger meaning tying together these eccentric set of stories. None of the stories are really "about" anything. They certainly are not about hit-men pontificating about burgers. Nor is the film really a satire or a farce, although it contains elements of both. At times, it feels like a tale that didn\'t need to be told, but for whatever reason this movie tells it and does a better job than most films of its kind, or of any other kind.',
  'I like the bit with the cheeseburger. It makes me want to go and get a cheeseburger',
  "Possibly the most influential movie made in history since the first movie ever made .Even after 25 years and a countless number of copy cats this movie absolutely holds up and feels fresh.\n\nAs many of you would know this movie is sort of like a tribute to the pulp stories written back in the 40s and 50s which have punchy and witty dialogue and over the top violence. The random chit chat between characters regarding obscure things might feel a waste of time as it doesn't move the plot. But Tarantino is a genius and he believes in character development more than plot development so these chats actually humanizes the characters a lot and makes them relatable and memorable(The absolute converse of Nolan). This movie is just a collection of wonderful scenes back to back from start to finish.. The movie is a must watch just for the awesome screenplay alone. To top it off all the actors arguably give their career best performance(think about it ,it is true) . And the ability of Tarantino to just create a really tense or ridiculous situation out of nowhere(like the psycho tribute scene with marsellus and butch which is so unexpected or that adrenaline shot scene) is just awesome and keeps you on the edge of your seats .\n\nSo if you want an entertaining but clever movie this is the one you are looking for . It is funny, filled with some of the best dialogues ever ,superbly acted, great soundtracks(I just wanna see Tarantino's ipod , boy does he have great taste) and is just a spectacular experience.\n\nSpoilers ahead(duh): For those of you who say the characters are hollow I suggest you to revisit the gold watch sequence. Walken's Capt Koons tells a story to young butch about how Butch's grandpa requested an enemy soldier to return his watch to his family and the soldier agreed. Similarly Butch could have left Marsellus but the legacy of his watch made him save him which I believe was a nice touch. And how Jules(one of my characters of all time) undergoes a change is also a noteworthy counter argument to the above mentioned criticism.\n\nSorry for the huge review..",
  "This is Tarantino's masterpiece, there's no other way to say it. It has arguably one of the smartest scripts I've ever seen. The story, which is non-linear, is so well constructed it takes several viewings to grasp it all. The movie doesn't seem to be about any spesific thing, but there is a subtle hint of redemption as a central theme. The characters and preformances in this movie are practically perfect. This is still one of the best performances I've seen from Sam Jackson, and it's an outrage he didn't win an Oscar. Each scene has its own unique flavour and charm, every segment has its own arc while also tying into the main plot. The comedy is great, the serious moments are great, every word of dialogue is exciting despite seemingly not having any reason to exist. This movie is just such a great time, and I recommend it to everyone who loves movies. I cannot think of a single genuine flaw with it, and it will remain one of my favorite movies for a long time.",
  'I can only speak for myself, but I had never seen anything as stylish, cleverly constructed, well written and electrifying as this milestone when I first saw it in 1994. What really pulled me in right from the start is what we\'ve now come to know as a Tarantino trademark: the dialogue. When gangsters Jules and Vincent talk to each other (or all the other characters, for that matter) there is a natural flow, a sense of realism and yet something slightly over the top and very theatrical about their lines – it\'s a mixture that immediately grabs your attention (even if it\'s just two dudes talking about what kind of hamburger they prefer, or contemplating the value of a foot-massage). Then there\'s the music: the songs Tarantino chose for his masterpiece fit their respective scenes so perfectly that most of those pieces of music are now immediately associated with \'Pulp Fiction\'. And the narrative: the different story lines that come together, the elegantly used flashbacks, the use of "chapters" – there is so much playful creativity at play here, it\'s just a pure joy to watch.\n\nIf you\'re a bit of a film geek, you realize how much knowledge about film and love for the work of other greats – and inspiration from them - went into this (Leone, DePalma, Scorsese and, of course, dozens of hyper-stylized Asian gangster flicks), but to those accusing Tarantino of copying or even "stealing" from other film-makers I can only say: There has never been an artist who adored his kind of art that was NOT inspired or influenced by his favorite artists. And if you watch Tarantino\'s masterpiece today, it\'s impossible not to recognize just what a breath of fresh air it was (still is, actually). Somehow, movies - especially gangster films - never looked quite the same after \'Pulp Fiction\'. Probably the most influential film of the last 20 years, it\'s got simply everything: amazing performances (especially Sam Jackson); it features some of the most sizzling, iconic dialogue ever written; it has arguably one of the best non-original soundtracks ever - it\'s such a crazy, cool, inspirational ride that you feel dizzy after watching it for the first time. It\'s – well: it\'s \'Pulp Fiction\'. 10 stars out of 10.\n\nFavorite films: http://www.IMDb.com/list/mkjOKvqlSBs/\n\nLesser-known Masterpieces: http://www.imdb.com/list/ls070242495/\n\nFavorite TV-Shows reviewed: http://www.imdb.com/list/ls075552387/',
  "Pulp Fiction is the most original, rule breaking film I have ever seen. Instead of following the widely used 3 act structure, Pulp Fiction makes up its own and while the 3 stories may seem completely disconnected at first, once you look closely you can find the underlying themes that they all share. Anyone who says that the movie lacks focus or has no meaning hasn't analysed enough. I highly recommend this film since it is number one on my list of my favourite movies of all time.",
  'Before I saw this I assumed it was probably overrated. I was wrong. It lives up to and surpasses its reputation in pretty much every way. I would definitely recommend.',
  "To put this in context, I am 34 years old and I have to say that this is the best film I have seen without doubt and I don't expect it will be beaten as far as I am concerned. Obviously times move on, and I acknowledge that due to its violence and one particularly uncomfortable scene this film is not for everyone, but I still remember watching it for the first time, and it blew me away. Anyone who watches it now has to remember that it actually changed the history of cinema. In context- it followed a decade or more of action films that always ended with a chase sequence where the hero saved the day - you could have written those films yourself. Pulp had you gripped and credited the audience with intelligence. There is not a line of wasted dialogue and the movie incorporates a number of complexities that are not immediately obvious. It also resurrected the career of Grease icon John Travolta and highlighted the acting talent of Samuel L Jackson. There are many films now that are edited out of sequence and have multiple plots etc but this is the one they all want to be, or all want to beat, but never will.",
  'Just the best movie... I can imagine my family seeing this movie in 30 years. I really love this movie and his soundtrack.',
  'If you think "Pulp Fiction" is brilliant, you\'re wrong. It\'s more than that. It\'s a milestone in the history of film making. It\'s already a classic. But why? Because of the many "f" words, or maybe because of the brain and skull pieces on the rear window of a car? No, that\'s surely not the point (unfortunately some other users - fortunately the minority - don\'t get it). Tarantino has made a movie that\'s someway different from many other action, gangster or crime movies. What\'s so different? He knows the subject of the movie is "cool", he knows it\'s a product of mass culture, and he even likes it by himself. But he smiles at it and tells three great stories with a lot of irony. And this irony is the first point. The second point is that he gave souls to extremely schematic characters. They surely aren\'t another action heroes who you forget as fast as you can twinkle. They are human beings like we are, talking about Burger King and McDonalds, about TV series and a foot massage. They just earn their money with killing others or selling drugs. What else is so great about "Pulp Fiction"? It\'s the acting, the directing, the cinematography, the soundtrack, the sense of humour and the whole rest. In my opinion it\'s all worth nothing less than a 10 out of 10. A masterpiece.',
  "I don't get as much out of Pulp Fiction as everybody else does. I don't like that it doesn't have a strong message, I think the dialogue goes on self-impressed tangents some times and the Mia Wallace story doesn't tie in with the rest of the film and it just isn't that interesting to me. However, that doesn't mean I don't appreciate its style. The entire film from the costume design, to the camera work, to the setting of LA are dripping with noir atmosphere. The writing, while it does jerk itself off at times, is still pretty memorable and enjoyable to listen to. The acting is fantastic all around, getting especially good performances from Samuel L Jackson, John Travolta and Bruce Willis.\n\nWhile I don't think its a masterpiece, it is something fun to watch if you're looking for something different and intriguing.",
  'My oh my.  "Pulp Fiction" is one of those roller-coasters of a movie.  It is both a joy and a trial to sit through.  Amazingly original and unforgettable, Quentin Tarantino\'s trash masterpiece never gets old or seem outdated.  It put a face on American independent film making in 1994. Miramax had been around since the 1970s and no one had heard of it before this film.  Studios went into a panic when this film came out because they knew it would be an amazing hit.  Of course it was.  Independent film making became the rage and hit its peak in 1996 when four of the five nominated Best Picture films were from independent studios.  The screenplay and direction by Tarantino are quite amazing, but the cast makes the film work. John Travolta (Oscar nominated) re-invented his career with this film. Bruce Willis cemented his celebrity.  Samuel L. Jackson and Uma Thurman (both Oscar nominees) became marketable superstars.  Others who make appearances include: Ving Rhames, Christopher Walken, Eric Stoltz, Rosanna Arquette, Steve Buscemi, Frank Whaley, Harvey Keitel, and of course Quentin Tarantino himself.  They all leave lasting impressions as well.  Samuel L. Jackson stood out the most to me, his lack of substantial screen time may have cost him the Oscar.  Just an amazing accomplishment, all involved deserve recognition.  Easily 5 stars out of 5.',
  "I don't understand why this is the 8th highest rated movie of all time. I saw how highly rated it was so decided to watch it. After the first half an hour I found it enjoyable but my mind wasn't blown. As the movie continued I kept waiting for something to happen that warrants its 8.9 stars. Nothing actually happens in this movie. I hope its just me because through my eyes its the most overrated movie of all time.",
  'Perhaps before I start talking about why I think Pulp Fiction works on so many levels, I should mention briefly how I came to watch the film. Pulp Fiction was a film I had heard of at a very young age, and I\'m not quite sure why. I had also wanted to see it for a very long time- again, I\'m not quite sure why. Maybe the mere fact that it was R-rated and notorious for its violence perked the interest of my 14-year-old self. Nevertheless, I got round to watching it about a month before my 15th birthday- incredibly late at night on an occasion where I found myself home alone. I recall being blown away by the film, but also somewhat overwhelmed and confused. I had never seen anything like it before, and walked away from it being kinda sure I\'d enjoyed it. I couldn\'t say for sure though.\n\nStill feeling curious about the film, and the many mysteries relating to it that I\'d failed to determine on my first viewing, I re-watched the film just a couple of months later. And something just clicked for me- I fell in love with the film. Everything about the movie suddenly worked for me, and I found the second viewing to be perhaps the shortest two and a half hours of my life I\'ve ever experienced. My own sense of time was warped and bended to the extent that it was in the film itself. That very night it somehow became one of the greatest things I had ever watched; one of the only films I\'d seen that I barely hesitated to call a masterpiece.\n\nSince then I\'ve watched it another six or seven times- almost once a year- and I continue to feel motivated to watch it because I honestly feel like I get something new out of it each time I watch it.\n\nSpoilers ahead, by the way. I plan to get fairly in-depth with my review of this movie, so just a warning for those that haven\'t seen it.\n\nOne of the great things about Pulp Fiction is its refusal to fall under any particular genre or category of movies. It feels like a comedy when Jules and Vince discuss fast food and foot massages for nearly ten minutes It feels like a thriller during several segments, notably the scene where Mia overdoses on heroin and the climactic Mexican stand-off in the restaurant- two sequences that would have even Alfred Hitchcock on the edge of his seat. The crime genre is represented through the character\'s actions- most of the cast are criminals in one way or another, whether they murder, steal, or take generous helpings of class-A drugs. The infamous gimp scene feels straight out of a horror movie.\n\nThis collage of various movie genres is one of the things that makes the movie stand out- by themselves, certain scenes may feel familiar, but when they\'re all blended together so well like they are in Pulp Fiction, the end product ends up feeling unique. This is true for most of Quentin Tarantino\'s films- he borrows elements from different genres, and homages/ references dozens of older movies in order to create something that feels unique, even if most of the individual elements themselves aren\'t too original. I can see why some people may not be a fan of the fact Tarantino essentially steals from the lesser known works of those who came before him, but for me, I love it- I think he just makes it work due to the fact that his encyclopaedic knowledge of film allows him to borrow from so many sources. If he simply referenced about a half dozen or less films for each one of his movies, then I think that would start to feel like plagiarism.\n\nNow, I could go on about the acting, the screenplay, the direction, and the glorious soundtrack, but really, what\'s there to say about these elements of the film that haven\'t already been said? Tarantino\'s screenplay is one of the most acclaimed and quoted from the past couple of decades, and deservedly won him his first Oscar. His direction has been similarly praised, and the soundtrack has become iconic- most impressive of all is that Tarantino chose music that goes so well with the images they accompany. And yes, the acting is phenomenal- the film features what is almost certainly Samuel L. Jackson\'s best performance, one of Bruce Willis\' most interesting performances of his long and successful career, and some great work from Uma Thurman. Even the supporting actors are memorable, including Harvey Keitel, Tim Roth, and Ving Rhames as Marsellus Wallace (who\'s arguably the film\'s main character- think about it; without him, none of the three main stories would exist- Butch ripped off Marsellus Wallace, Vince took out Wallace\'s girlfriend on a date that eventually went horribly wrong, and Vince and Jules were assigned to retrieve the mysterious briefcase for Wallace). And of course, who can forget that Pulp Fiction single-handedly made John Travolta cool again- an absolutely monumental achievement. Of course Travolta did eventually succeed in making himself a joke again a few years later when he made Battlefield Earth, but that\'s another story (or perhaps better left for another review)?\n\nAs I said, I could indulge in commenting on these areas of the film, but if you\'ve seen the movie you probably already know how good they are. Pulp Fiction is also one of the most discussed film\'s of the past 20 years, so you\'ve likely already come across reviews or rabid Tarantino fans who\'ve gushed about why the film works so well, and how fantastic all the various components of the film are.\n\nSo instead, I\'m going to backtrack back to my point about seeing something new in the film upon every new viewing, and explain what I took away from the movie on my most recent viewing.\n\nPulp Fiction was always controversial for its violence, with some criticising its depiction of assaults, shootings, beatings, and exploding heads. But for me, I found the film to have an almost anti-violence message of sorts, and I only realised this on my most recent viewing. Now bear with me, because I know that sounds like a somewhat ridiculous claim, but I have my reasons. Pulp Fiction may be violent, but it doesn\'t promote violence. Sure, the violence may be somewhat stylised and at times over the top, but that doesn\'t mean the film is saying that violence is something trivial. Violent acts in the film are often shown to have consequences for those that commit them. Vince\'s carelessness with his gun- an instrument of violence- causes Marvin\'s head to be blown off in the backseat of Jules\' car, which leads to a near twenty minute detour in which they must take the car to Jimmy\'s house and consult "The Wolf" to assist them in cleaning the car and disposing of the gory evidence. Marsellus\' desire to get revenge on Butch by presumably killing him leads them both to the basement of the rednecks- indeed, they are both fighting each other in the redneck\'s store when they are taken captive.\n\nBut in the climax of the film, where guns are being pointed at several characters in the middle of a tense Mexican stand-off, not a single bullet is fired. One would expect the tension to be eventually broken and the bullets to start flying, especially if one has seen some of Taratino\'s other movies that end in explosive and violent climaxes (see Reservoir Dogs, Inglorious Basterds, Death Proof, Django Unchained, and Kill Bill Volume 1). But instead, Jules, who we\'ve seen to be ruthless and unafraid to kill earlier in the movie, believes himself to be a changed man after miraculously surviving a hail-storm of bullets from a criminal that looks a little like Jerry Seinfeld and so instead decides to defuse the situation peacefully. He talks down Pumpkin and Honey-Bunny from killing or robbing anyone, and insists that his dim-witted associate Vince refrain from hurting anyone too. He delivers an absolutely stunning monologue where he ponders the bible reading he quoted so confidently earlier in the movie, and it\'s damn-near poetic. It sends a shiver up my spine every time I watch that scene, and indeed, Jules\' speech works. The two robbers get up after Jules gives them some money from his famous wallet, and then leave. Jules and Vince do the same a few moments later.\n\nNot a shot is fired. No one is killed. It\'s an absolutely beautiful scene.\n\nIt may have been tempting for Tarantino to give the film a "Wild Bunch-esque" ending, but instead he refrained, and I\'m glad he did so. The climax to Pulp Fiction is absolutely stunning- fifteen minutes of tension, almost poetic dialogue, and brilliant acting, especially from Samuel L. Jackson. And it took me seven viewings just to notice how beautifully peaceful the ending of the film was. And that\'s why it\'s my favourite film of all time- I get new meaning from it, or appreciate different areas of it every-time I watch it.\n\nIf you\'ve never seen Pulp Fiction before, I implore you to go watch it. And if you have seen Pulp Fiction before, I implore you to go watch it again.',
  "Viewers are taken on a ride through three different stories that entertwine together around the world of Marcellus Wallace. Quentin Tarantino proves that he is the master of witty dialogue and a fast plot that doesn't allow the viewer a moment of boredom or rest. From the story of two hit-man on a job, to a fixed boxing match to a date between a hit-man and the wife of a mob boss. There was definitely a lot of care into the writing of the script, as everything no matter the order it is in, fits with the story. Many mysteries have been left such as what is inside of the briefcase and why Marcellus Wallace has a band-aid on the back of his neck, which may be connected. The movie redefined the action genre and reinvigorated the careers of both John Travolta and Bruce Willis. This movie is required viewing for any fan of film.",
  "Pulp Fiction may be the single best film ever made, and quite appropriately it is by one of the most creative directors of all time, Quentin Tarantino. This movie is amazing from the beginning definition of pulp to the end credits and boasts one of the best casts ever assembled with the likes of Bruce Willis, Samuel L. Jackson, John Travolta, Uma Thurman, Harvey Keitel, Tim Roth and Christopher Walken. The dialog is surprisingly humorous for this type of film, and I think that's what has made it so successful. Wrongfully denied the many Oscars it was nominated for, Pulp Fiction is by far the best film of the 90s and no Tarantino film has surpassed the quality of this movie (although Kill Bill came close). As far as I'm concerned this is the top film of all-time and definitely deserves a watch if you haven't seen it.",
  "This is my favorite film of all time. Every second of this film is engaging, and I'm not exaggerating when I say that. Tarantino's direction and script is brilliant, and every role is perfectly cast. Everything that happens in this movie has a purpose, and you don't realize it's hidden in plain sight until the final moments",
  "It took fifteen years and a subscription to Netflix to finally get around to seeing this film. It was well worth the wait. If all one were to see is the byplay between Samuel L. Jackson and John Travolta, it would be worth the price of admission. But this is only part of the incredible effect this film has. I don't even like gore in movies; I avoid it. But Tarentino weaves a culture of violence where there is actual humanity. We care about these bad guys. There is scene after scene of people being pushed to the limit. The very idea that the boxer, played by Bruce Willis, would risk everything to retrieve a watch that has been transported on two occasions, shove up someone's ass, is amazing. And very, very funny. I can't begin to list all the wonderful scenes. The opening dialogue is incredible as Trovolta and Jackson are on their way to do a hit. They have virtually no respect for human life, yet they, themselves, are very human. The date scene with Uma Thurman with the adrenaline shot. The crazy's in the pawn shop. And, the effort to clean up the car after blowing away a kid in the back seat, the issue being that a guys wife would come home and be very unhappy to find a dead body and a blood filled car in the garage. It sounds horrible and makes me sound sick. I found Fargo to be a hilarious film as well. Is there something wrong with me?",
  "One of the best movies I have ever seen is a fun and enjoyable story, as most of the film's characters were excellent, and it brought back the character of Butch and his girlfriend, Vincent was the best character and his death was shocking and I did not. Expect that, the interconnectedness of the film's stories was excellent, the direction in the film was excellent The film, the acting in the movie was very cool, I recommend it, and it is the best film by Quentin Tarantino",
  "I heard so many people claim that this movie is a masterpiece and it also has high ratings. So I was intrigued to give it a go. I didn't get much enjoyment out of it. Maybe you have to watch it multiple times to enjoy it. As for the first viewing it didn't satisfy me.",
  'Tarantino is without a doubt one of the best directors of all time and maybe the best of the 90\'s. His first film, Reservoir Dogs was amazing and claustrophobic, his segment in Four Rooms was by far the greatest (even though Rodriguez\'s was excellent too)and Jackie Brown is a wonderful homage to the Blaxploitation films of the 70\'s. However, Pulp Fiction remains my favourite.\n\nIt was nominated for so many Oscars that I still find it hard to believe that it only got one: Best original script. I\'m not complaining because Forrest Gump got best picture, since that film was also Oscar-worthy, but come on, movies like Tarantino\'s or the Shawshank Redemption deserved much more.\n\nAnyway, going back to the movie, I particularly liked the first and second chapters, and that\'s really a contradiction because one of the movie\'s finest characters, Mr. Wolf, appears on the third. Bruce Willis also does a great job, and as far as I\'m concerned he fell in love with the movie right after having read the script. I like the way his character gives a "tough guy" image at the beginning and then we discover he\'s so affectionate and tender to his wife. Travolta is obviously the star of the movie and his second encounter with Bruce Willis in the kitchen along with the scene where he dances with Uma Thurman is when the movie reaches it\'s highest point.\n\nThe other star is Samuel L. Jackson, who plays a wise assassin that obviously knows how to handle situations. "And I will strike down upon thee with great vengeance and furious anger..." is my favourite quote.\n\nSummarizing, Pulp Fiction is a modern classic and a must-see for anyone who is at least aware of what a movie is. I give it a 9 out of 10.',
  "I was tempted to stop watching it about four times. I went up to the end just to see if, at least, there is something extraordinary or unexpected there. I really can't understand what all these viewers found to this film, to score it so high. It's the first time (or the second I'm not sure!) that I get into the mood of writing a review. The reason, I guess, is because this film is the fifth in the top 250 ranking and I couldn't see why. The curiosity of the ranking was also the reason of watching it at the first place. There is a high end casting and a top level director but nothing more, IN MY OPINION. I tried looking it through the eyes of 1994 but still compared to other productions of the age is far behind. If you find yourself in the position wondering whether to watch this film or not, due to its place in the list\x85 don't waste your time!",
  'This movie is such a masterpiece for many reasons. The way everything connects makes it really important for you to pay attention. Lots of action and very funny. It also has a really good balance of everything you look for in a movie.',
  'Conflicted on this one. I like a lot of elements of it, the characters are (mostly) really compelling and interesting. I think some of the "Segments" are a lot stronger than others. I think that the first part of Bruce Willis\'s story is just... bland. I didn\'t find it very interesting or compelling. It gets interesting though. The first and last are by far my favourite, but for the most part it\'s pretty strong. I do think this film is slightly overrated. That\'s not to say that it isn\'t good, but I think a lot of people think it is a whole lot better then it is. The film is acted well and I enjoyed all the action, not to mention the comedy in this film caught me by surprise. I found it really hilarious. I see why everyone had/has a crush on Uma Thurman now, she really is deserving of the main character on the cover. I also remembered how much I like long shots. I think they allow exposition to be delivered in a way that is more pleasing to the audience. Overall, good crime flick with some great comedy and action but a tad overrated. 7-8/10.',
  "It's not a film it's a experience. The characters are very memorable The story of nonlinear dialogue and action no the whole movie is fast and fun there's no down time or anything boring the movie is long. There's more then 10 memorable scenes that is referenced in any type of media. I like to think of pulp fiction as resvoir dogs another film by the genius tarrintino as a better version of it. I thought resvoir dogs had the coffe scene at the start and mr white hurting the police officer. I thought the movie was stretched out more then it needed to and the ending gave me such a poor taste in my mouth. I still like it, like I would get a poster or t shirt of resvoir dogs to show my opinion. There did you see that I lost focus that's what pulp fiction does it makes you forget about the world but remember about the experience. The scene where butch reclaims himself in the eyes of walleces is a redemption story in itself about how we can all get together without fighting and being greedy. The part where 2 scenes ( tiny bit of a spoiler but watch Pulp fiction now if you haven't) where somehow at point blank range a revolver didn't shoot Jules and vincent and somehow Vincent shoots one of the guys in the head on accident. Thats a message life is like a box of chocolates the things you never expect will happen. Another part of the idea is the violence and sex violence plays a big part in this movie these are horrible people that everyone loves when I see this film I feel like it's a LEGO set controlled by tarrintino everything goes together like cause and effect yet while still being random sex is not that big in the film expect for the epic scene where butch remdeems himself and when butch is making out with his girlfriend but that scene was still great because he had to find his watch holy crap. This movie has dialogue lots of it and that's fantastic he is known for dialogue the tarrintino himself said that dialogue is a major component in his films and it shows, other films have such poopy dialogue where all they do is talk about nothing like get on with it. Pulp fiction is the story of 3 as 1 and should be hands down the number 1 film of the millennium ."]
sample_data = {'id': 'tt0110912',
 'plot': 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
 'reviews': reviews,
'image': 'https://m.media-amazon.com/images/M/MV5BYTViYTE3ZGQtNDBlMC00ZTAyLTkyODMtZGRiZDg0MjA2YThkXkEyXkFqcGc@._V1_.jpg',
 'name': 'Pulp Fiction',
 'country': 'US',
 'release_date': '14/10/1994',
 'ratings': 8.9,
 'total_votes': 2273320,
    'positive_reviews':['asdf'],
    'negative_reviews':['asdf'],
    'positive_review_summary': 'asdf',
    'negative_review_summary': 'asdf'}
movie_details = classify_summarize_reviews(sample_data)
print(movie_details)
'''