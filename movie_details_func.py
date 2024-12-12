import requests
import time
from bs4 import BeautifulSoup
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

HEADERS = {
	"x-rapidapi-key": "d4f11cda99mshaf0c893d26831e1p1e08ecjsnf73b7a997619",
	"x-rapidapi-host": "online-movie-database.p.rapidapi.com"
    }

resp = None
movie_name = []

def scrape_reviews(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    reviews_soup = soup.find_all('drawer-more')
    reviews = []
    for i in reviews_soup:
        # Get the raw text, strip surrounding whitespace, and replace multiple spaces/newlines with a single space
        clean_text = re.sub(r'\s+', ' ', i.text.strip())
        clean_text = re.sub(r'\b(show less|show more)\b', '', clean_text, flags=re.IGNORECASE)
        reviews.append(clean_text)
    return reviews

def perform_clicks(driver):
    num_clicks = 5
    for _ in range(num_clicks):
        try:
            # Wait until the "Load More" button is clickable and then click it
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'rt-button[data-qa="load-more-btn"]'))
            )
            load_more_button.click()
            time.sleep(2)  # Wait for new reviews to load (adjust based on load time)
        except Exception as e:
            print("Error clicking Load More button:", e)
            break

def get_link(html_source):
    soup = BeautifulSoup(html_source, "html.parser")
    link = soup.find("search-page-media-row", {"data-qa": "data-row"})
    link = soup.find("a", {"data-qa": "info-name"})
    href = link.get("href") if link else None
    return href

def get_movie_site(movie_name, driver):
    q = movie_name.lower().replace(" ", "%20")
    url = f"https://www.rottentomatoes.com/search?search={q}"
    driver.get(url)
    page_html = driver.page_source
    return get_link(page_html)

def get_movie_reviews(movie_name):
    try:
        driver = webdriver.Chrome()
        url = get_movie_site(movie_name, driver) + "/reviews?type=user"
        driver.get(url)
        perform_clicks(driver)
        page_html = driver.page_source
        driver.quit()
        reviews = scrape_reviews(page_html)
        return reviews
    except:
        return []

def api_auto_complete(query):
    global resp
    global movie_name
    metacritic_api_url = "https://online-movie-database.p.rapidapi.com/auto-complete"
    querystring = {"q":query}
    response = requests.get(metacritic_api_url, headers=HEADERS, params=querystring)
    response = response.json()
    resp = response
    movie_name = []
    for i in response['d']:
        movie_name.append(i['l'])
    return movie_name

def get_movie_plot(movie_id):
    plot_url = "https://online-movie-database.p.rapidapi.com/title/v2/get-plot"
    querystring = {"tconst":movie_id}
    response = requests.get(plot_url, headers=HEADERS, params=querystring)
    response = response.json()
    try:
        val = response['data']['title']['plot']['plotText']['plainText']
        return val
    except:
        return ""
    
def parse_ratings_res(response):
    try:
        country = response['data']['title']['releaseDate']['country']['id']
        day = response['data']['title']['releaseDate']['day']
        month = response['data']['title']['releaseDate']['month']
        year = response['data']['title']['releaseDate']['year']
        release_date = f"{day}/{month}/{year}"
        ratings = response['data']['title']['ratingsSummary']['aggregateRating']
        total_votes = response['data']['title']['ratingsSummary']['voteCount']
        return {'country': country, 'release_date': release_date, 'ratings': ratings, 'total_votes':total_votes}
    except:
        return ""
    
def get_movie_ratings(movie_id):
    rating_url = "https://online-movie-database.p.rapidapi.com/title/v2/get-ratings"
    querystring = {"tconst":movie_id}
    response = requests.get(rating_url, headers=HEADERS, params=querystring)
    response = response.json()
    return parse_ratings_res(response)

def get_revs_res(response):
    reviews = []
    try:
        for i in response['reviews']:
            reviews.append(i['reviewText'])
        return reviews
    except:
        return []

def get_movie_reviews_critic(movie_id):
    reviews_url = "https://online-movie-database.p.rapidapi.com/title/get-user-reviews"
    querystring = {"tconst":movie_id}
    response = requests.get(reviews_url, headers=HEADERS, params=querystring)
    response = response.json()
    return get_revs_res(response)

def get_movie_id_critic(movie_name):
    global resp
    for i in resp['d']:
        if i['l'] == movie_name:
            return i

def get_movie_details(movie_name):
    id_dict = get_movie_id_critic(movie_name)
    img_url = id_dict['i']['imageUrl']
    name = id_dict['l']
    id = id_dict['id']
    plot = get_movie_plot(id)
    rating_dict = get_movie_ratings(id)
    reviews = get_movie_reviews_critic(id)
    movie_details = {'id': id, 'plot': plot, 'reviews':reviews, 'image':img_url, 'name':name}
    for i in rating_dict:
        movie_details[i] = rating_dict[i]
    return movie_details

def get_movie_details_overall(movie_name):
    temp = get_movie_details(movie_name)
    revs_list = get_movie_reviews(temp['name'].lower().replace(" ", "_"))
    temp['reviews'] += revs_list
    return temp