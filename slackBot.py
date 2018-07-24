import os
import time
import re
from slackclient import SlackClient
import datetime
import csv


# export SLACK_BOT_TOKEN='your bot user access token here'
# slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

slack_client = SlackClient('xoxb-397785387536-402566650723-MKUz9KcecNGBx7jqP8M3vSSt')

ai_bot_id = None

# constants
RTM_READ_DELAY = 1
COMMAND = "Log"

def getCsvData():
    data = []
    with open('logs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i=0
        for row in csv_reader:
            data.append(int(float(row[0])))
            i+=1
    return data

def getTime(seconds):
    time = datetime.timedelta(seconds = int(seconds))
    return time

def parse_bot_commands(slack_events):
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            return event["text"], event["channel"]
    return None, None

def handle_command(command, channel, slack_logs):
    # Default response is help text for the user
    response = "Not sure what you mean. Try *" + COMMAND + "*."

    # This is where you start to implement more commands!

    # For logs:
    # [0] Generation
    # [1] Highest Score
    # [2] Current Score
    # [3] Games Played
    # [4] Start Time

    print(slack_logs)
    currentTime = time.time()
    timeSinceStart = getTime(currentTime - slack_logs[4])

    if command.startswith(COMMAND):
        response = "Generation: " + str(slack_logs[0]) + " Highest Score: " + str(slack_logs[1]) + " Current Score: " + str(slack_logs[2]) + " Games Played: " + str(slack_logs[3]) + " Time Since Start: " + str(timeSinceStart)

    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response or default_response
    )

def reconnectLoop(slack_client, channel, connectionLostTime):
    connected=False
    while not connected:
        print("Attempting reconnection ...")
        try:
            connected = slack_client.rtm_connect(with_team_state=False)
        except:
            print("... failed")
        time.sleep(10*RTM_READ_DELAY)
    print("Reconnected")
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text="Connection lost at "+connectionLostTime+". Reconnected now."
    )
    
    

# if __name__ == "__main__":
if slack_client.rtm_connect(with_team_state=False):
    print(datetime.datetime.now())
    print("AI Bot connected and running!")
    # Read bot's user ID by calling Web API method `auth.test`
    ai_bot_id = slack_client.api_call("auth.test")["user_id"]
    channelId="CBUFRK2EQ"
    while True:
        try:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                slack_logs = getCsvData()
                handle_command(command, channel, slack_logs)
        except:
            connectionLostTime = str(datetime.datetime.now())
            print ("Connection failed at ", connectionLostTime)
            reconnectLoop(slack_client, channelId, connectionLostTime)
        finally:
            time.sleep(RTM_READ_DELAY)
else:
    print("Connection failed from the start. Quitting...")
