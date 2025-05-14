from twilio.rest import Client
import os

def send_twilio_message():
    # Twilio credentials from environment variables
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_FROM_NUMBER')
    to_number = os.getenv('TWILIO_TO_NUMBER')

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body="This is a scheduled message from your cron job!",
        from_=from_number,
        to=to_number
    )

    print(f"Message sent: {message.sid}")

if __name__ == "__main__":
    send_twilio_message()
