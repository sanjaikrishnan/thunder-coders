import vonage 
client = vonage.Client(key="12eaf3e4", secret="AdSYVY7CLEi0xsCA")
sms = vonage.Sms(client)
responseData = sms.send_message(
    {
        "from": "Vonage APIs",
        "to": "+919047114805",
        "text": "happy birthday sanjai",
    }
)

if responseData["messages"][0]["status"] == "0":
    print("Message sent successfully.")
else:
    print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
