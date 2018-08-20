import csv

def write_csv(index, data):
    slack_logs[index] = data

    # For slack_logs:
    # [0] Scores
    # [1] All Time HS
    # [2] Start Time
    # [3] Games Played

    with open("logs.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(slack_logs)

def write_scores(data):
    df = pd.DataFrame(data)
    df.to_csv('scores.csv')
