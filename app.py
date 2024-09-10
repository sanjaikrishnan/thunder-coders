from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import vonage

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Vonage API credentials

client = vonage.Client(key="12eaf3e4", secret="AdSYVY7CLEi0xsCA")
sms = vonage.Sms(client)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_sms', methods=['POST'])
def send_sms():
    data = request.get_json()  # Get the JSON data from the frontend
    latitude = data['latitude']
    longitude = data['longitude']

    # Create Google Maps link using the latitude and longitude
    google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"

    # Send SMS with location and Google Maps link
    responseData = sms.send_message(
        {
            "from": "YourAppName",
            "to": "+919047114805",  # Replace with recipient's phone number
            "text": f"Alert: A lone woman detected. Location: Lat {latitude}, Long {longitude}. View on map: {google_maps_link}",
        }
    )
    
    print(responseData)  # Debugging: Print the response from the Vonage API

    if responseData["messages"][0]["status"] == "0":
        return jsonify({'status': 'success', 'message': 'Message sent successfully.'})
    else:
        error_message = responseData['messages'][0]['error-text']
        print(f"Message failed with error: {error_message}")
        return jsonify({'status': 'error', 'message': error_message}), 400

if __name__ == '__main__':
    app.run(debug=False)
