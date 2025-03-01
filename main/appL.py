from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

YOUTUBE_API_KEY = "--"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

@app.route('/youtube_videos', methods=['GET'])
def get_youtube_videos():
    search_query = request.args.get("query", "Python full course")  # Default to "Python full course"
    
    params = {
        "part": "snippet",
        "q": search_query,  # Search for specific courses
        "type": "video",
        "maxResults": 10,  # Fetch 10 videos
        "videoDuration": "long",  # Filter for long videos (courses)
        "key": YOUTUBE_API_KEY
    }
    
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
