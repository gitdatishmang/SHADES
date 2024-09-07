from twilio.rest import Client

account_sid = 'KEY'
auth_token = 'KEY'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='whatsapp:Number',
  body='This is how I messaged my phone for my glasses to read',
  to='whatsapp:Number'
)

print(message.sid)