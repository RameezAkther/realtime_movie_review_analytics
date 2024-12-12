from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import couchdb
import hashlib
import movie_details_func
import sentiment_analysis

app = Flask(__name__)
app.secret_key = "your_secret_key"  # For session management

# CouchDB connection
couch = couchdb.Server('http://admin:password@127.0.0.1:5984/')

if 'users' in couch:
    db = couch["users"]
else:
    db = couch.create("users")
db_2 = couch['movies']

sample_data = {'id': 'tt0110912',
 'plot': 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
 'reviews': ["asdf", "asdf"],
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


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def search_movie_titles(query, db, limit=10):
    """
    Search for movie titles in CouchDB that start with the specified query string.
    
    Parameters:
    - query (str): The search term to match the start of movie titles.
    - db (Database): The CouchDB database instance.
    - limit (int): The maximum number of results to return.
    
    Returns:
    - List of matching movie titles.
    """
    # Ensure query is a string
    query = str(query)
    
    # Query the view, starting with the search term and limiting results
    result = db.view(
        'movie_titles/by_title',
        startkey=query,
        endkey=query + '\ufff0',  # Ensures partial matching for CouchDB
        limit=limit
    )
    
    # Extract and return movie titles from the result
    titles = [row.key for row in result]
    return titles

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Check if user already exists
        if username in db:
            flash("Username already exists. Try logging in.", "danger")
            return redirect(url_for("signup"))

        # Hash the password and save to CouchDB
        db[username] = {"username": username, "password": hash_password(password)}
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))
    
    return render_template("signup.html", page="signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Check if user exists and password matches
        if username in db and db[username]["password"] == hash_password(password):
            session["username"] = username
            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    
    return render_template("signup.html", page="login")

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    suggestions = movie_details_func.api_auto_complete(query)
    return jsonify(suggestions=suggestions)

@app.route("/dashboard")
def dashboard():
    if "username" in session:
        return render_template("search.html")
    else:
        return redirect(url_for("login"))
    
@app.route('/movie')
def movie_page():
    if "username" not in session:
        return redirect(url_for("login"))

    movie_title = request.args.get('title', '')
    if movie_title:
        # Get the user's document from CouchDB
        user_doc = db.get(session["username"])
        
        # Search for the movie in the user's history
        movie_details = next((movie for movie in user_doc.get("history", []) if movie["name"] == movie_title), None)
        
        if not movie_details:
            movie_details = movie_details_func.get_movie_details_overall(movie_title)
            movie_details = sentiment_analysis.classify_summarize_reviews(movie_details)

            if movie_details:
                user_doc.setdefault("history", []).append(movie_details)
                db.save(user_doc)
            else:
                flash("Movie details not found.", "warning")
                return redirect(url_for("dashboard"))
        
        return render_template('movie_details.html', movie=movie_details)
    else:
        flash("No movie title provided.", "warning")
        return redirect(url_for("dashboard"))



@app.route("/history")
def history():
    if "username" in session:
        # Fetch the user's history from CouchDB
        user_doc = db.get(session["username"])
        movie_history = user_doc.get("history", [])
        return render_template("history.html", history=movie_history)
    else:
        return redirect(url_for("login"))
    
@app.route("/delete_history", methods=["POST"])
def delete_history():
    if "username" in session:
        movie_id = request.form["movie_id"]
        user_doc = db.get(session["username"])
        
        # Remove the movie with the specified ID
        user_doc["history"] = [movie for movie in user_doc.get("history", []) if movie["id"] != movie_id]
        
        # Save updated history
        db.save(user_doc)
        
        flash("Movie removed from history.", "info")
        return redirect(url_for("history"))
    else:
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
