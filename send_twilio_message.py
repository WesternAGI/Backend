from twilio.rest import Client
import os

def send_twilio_message():
    # Twilio credentials from environment variables
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_FROM_NUMBER')
    to_number = os.getenv('TWILIO_TO_NUMBER')

    print("Twilio credentials loaded successfully.")
    print(f"Account SID: {account_sid}")
    print(f"Auth Token: {auth_token}")
    print(f"From Number: {from_number}")
    print(f"To Number: {to_number}")
    
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body="This is a scheduled message from your cron job!",
        from_=from_number,
        to=to_number
    )

    print(f"Scheduled message sent to Twilio number: {to_number}")


if __name__ == "__main__":
    send_twilio_message()
