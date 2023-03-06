from twilio.rest import Client

# Set environment variables for your credentials
account_sid = "AC02cefd407ce832492d7d9e4418366db0"
auth_token = "273740116cc165550a194ce4ea91da4f"
client = Client(account_sid, auth_token)
message = client.messages.create(
  body="Hello from Twilio",
  from_="+12706790926",
  to="+34652034697"
)
print(message.sid)