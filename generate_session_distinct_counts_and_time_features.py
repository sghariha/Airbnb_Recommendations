import csv
from collections import defaultdict
import numpy as np
from scipy import stats


#----------------- ugly functions for generating features ------------------#

# for processing sessions.csv to get og data
def read_csv(fname):
    features = dict()
    data = []

    with open(fname, encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)
        headers = next(csvreader)
        for i, header in enumerate(headers):
            features[header] = i
        for line in csvreader:
            data.append(line)
    return features, data


# adds features: total sessions and number of unique
#     actions, action_types, action_details, and device_types
def add_all_and_distinct_counts_to_user_feature_vector(
        sessions_by_user, session_features, session_features_by_user):

    for user in session_features_by_user:
        user_n_actions = 0
        user_distinct_actions = set()
        user_distinct_action_types = set()
        user_distinct_action_detail = set()
        user_distinct_device_types = set()
        
        for entry in sessions_by_user[user]:
            user_distinct_actions.add(entry[session_features['action']])
            user_distinct_action_types.add(entry[session_features['action_type']])
            user_distinct_action_detail.add(entry[session_features['action_detail']])
            user_distinct_device_types.add(entry[session_features['device_type']])
            user_n_actions += 1
            
        session_features_by_user[user]['n_actions_per_user'] = user_n_actions
        session_features_by_user[user]['n_distinct_actions'] = len(user_distinct_actions)
        session_features_by_user[user]['n_distinct_action_types'] = len(user_distinct_action_types)
        session_features_by_user[user]['n_distinct_action_detail'] = len(user_distinct_action_detail)
        session_features_by_user[user]['n_distinct_device_types'] = len(user_distinct_device_types)
        
    return session_features_by_user


# adds features: total, average, std, max, min,
#     skew, and kurtosis for secs_elapsed
def add_timestamp_features_to_user_feature_vector(
        sessions_by_user, session_features, session_features_by_user):

    for user in session_features_by_user:
        
        user_seconds = []
        
        for entry in sessions_by_user[user]:
            seconds = entry[session_features['secs_elapsed']]
            if seconds == '':
                user_seconds.append(0.0)
            else:
                user_seconds.append(float(seconds))

        user_seconds = np.array(user_seconds)
            
        session_features_by_user[user]['total_seconds_elapsed'] = sum(user_seconds)
        session_features_by_user[user]['mean_seconds_elapsed'] = np.mean(user_seconds)
        session_features_by_user[user]['std_seconds_elapsed'] = np.std(user_seconds)
        session_features_by_user[user]['max_seconds_elapsed'] = max(user_seconds)
        if max(user_seconds) == 0:
            session_features_by_user[user]['min_seconds_elapsed'] = min(user_seconds)
        else:
            session_features_by_user[user]['min_seconds_elapsed'] = np.min(user_seconds[np.nonzero(user_seconds)])
        session_features_by_user[user]['skewness_seconds_elapsed'] = stats.skew(user_seconds)
        session_features_by_user[user]['kurtosis_seconds_elapsed'] = stats.kurtosis(user_seconds)
        
    return session_features_by_user


# adds features for ratio of time spent on each unique:
#     action, action_type, action_detail, and device_type
#
#     e.g. for each user:
#     <total time spent doing action X> : <total time spent>
#     <total time spent doing action_type X> : <total time spent>
#     <total time spent doing action_detail X> : <total time spent>
#     <total time spent on device_type X> : <total time spent>
def add_total_time_for_all_unique_sessions_things_to_user_feature_vector(
        sessions_by_user, session_features, session_features_by_user,
        distinct_actions, distinct_action_types, distinct_action_detail, distinct_device_types):

    for user in session_features_by_user:

        # initialize all to 0 since we won't encounter each case below

        for action in distinct_actions:
            action_total_time_name = 'action_%s_total_secs_elapsed' % action
            session_features_by_user[user][action_total_time_name] = 0.0

        for action_type in distinct_action_types:
            action_type_total_time_name = 'action_type_%s_total_secs_elapsed' % action_type
            session_features_by_user[user][action_type_total_time_name] = 0.0

        for action_detail in distinct_action_detail:
            action_detail_total_time_name = 'action_detail_%s_total_secs_elapsed' % action_detail
            session_features_by_user[user][action_detail_total_time_name] = 0.0

        for device_type in distinct_device_types:
            device_type_total_time_name = 'device_type_%s_total_secs_elapsed' % device_type
            session_features_by_user[user][device_type_total_time_name] = 0.0


        # add up time spent on each action, action_type, action_detail, and device_type

        user_seconds = []

        for entry in sessions_by_user[user]:
            seconds = entry[session_features['secs_elapsed']]
            if seconds == '':
                seconds = 0.0
            seconds = float(seconds)
            user_seconds.append(seconds)

            entry_action = entry[session_features['action']].replace(' ', '_')
            entry_action_type = entry[session_features['action_type']].replace(' ', '_')
            entry_action_detail = entry[session_features['action_detail']].replace(' ', '_')
            entry_device_type = entry[session_features['device_type']].replace(' ', '_')

            entry_action_total_time_name = 'action_%s_total_secs_elapsed' % entry_action
            entry_action_type_total_time_name = 'action_type_%s_total_secs_elapsed' % entry_action_type
            entry_action_detail_total_time_name = 'action_detail_%s_total_secs_elapsed' % entry_action_detail
            entry_device_type_total_time_name = 'device_type_%s_total_secs_elapsed' % entry_device_type

            session_features_by_user[user][entry_action_total_time_name] += seconds
            session_features_by_user[user][entry_action_type_total_time_name] += seconds
            session_features_by_user[user][entry_action_detail_total_time_name] += seconds
            session_features_by_user[user][entry_device_type_total_time_name] += seconds

        user_seconds = np.array(user_seconds)
        total_user_seconds = sum(user_seconds)


        # get ratio of time spend for each action, action_type, action_detail, device_type

        for action in distinct_actions:
            action_total_time_name = 'action_%s_total_secs_elapsed' % action
            action_ratio_time_name = 'action_%s_ratio_secs_elapsed' % action

            total_time_on_action = session_features_by_user[user][action_total_time_name]
            if total_time_on_action == 0.0:
                session_features_by_user[user][action_ratio_time_name] = 0.0
            else:
                session_features_by_user[user][action_ratio_time_name] = total_time_on_action / total_user_seconds

        for action_type in distinct_action_types:
            action_type_total_time_name = 'action_type_%s_total_secs_elapsed' % action_type
            action_type_ratio_time_name = 'action_type_%s_ratio_secs_elapsed' % action_type

            total_time_on_action_type = session_features_by_user[user][action_type_total_time_name]
            if total_time_on_action_type == 0.0:
                session_features_by_user[user][action_type_ratio_time_name] = 0.0
            else:
                session_features_by_user[user][action_type_ratio_time_name] = total_time_on_action_type / total_user_seconds

        for action_detail in distinct_action_detail:
            action_detail_total_time_name = 'action_detail_%s_total_secs_elapsed' % action_detail
            action_detail_ratio_time_name = 'action_detail_%s_ratio_secs_elapsed' % action_detail

            total_time_on_action_detail = session_features_by_user[user][action_detail_total_time_name]
            if total_time_on_action_detail == 0.0:
                session_features_by_user[user][action_detail_ratio_time_name] = 0.0
            else:
                session_features_by_user[user][action_detail_ratio_time_name] = total_time_on_action_detail / total_user_seconds

        for device_type in distinct_device_types:
            device_type_total_time_name = 'device_type_%s_total_secs_elapsed' % device_type
            device_type_ratio_time_name = 'device_type_%s_ratio_secs_elapsed' % device_type

            total_time_on_device_type = session_features_by_user[user][device_type_total_time_name]
            if total_time_on_device_type == 0.0:
                session_features_by_user[user][device_type_ratio_time_name] = 0.0
            else:
                session_features_by_user[user][device_type_ratio_time_name] = total_time_on_device_type / total_user_seconds

    return session_features_by_user


# write the engineered feature values per user to file
def write_engineered_session_csv(session_features_by_user, outfile_name):
    arbitrary_user = list(session_features_by_user.keys())[0]
    sorted_features = sorted(session_features_by_user[arbitrary_user].keys())

    outfile = open(outfile_name, 'w')

    # write header row with feature names
    outfile.write('id')
    for feature in sorted_features:
        outfile.write(',')
        outfile.write(feature)
    outfile.write('\n')

    # write feature values for each user
    for user in session_features_by_user:
        outfile.write(str(user))
        for feature in sorted_features:
            outfile.write(',')
            outfile.write(str(session_features_by_user[user][feature]))
        outfile.write('\n')
    outfile.close()


#------------------------------ main routine -------------------------------#

# get og session features and data from sessions.csv

print("Grabbing og sessions.csv data...")

session_data_filename = 'kaggle_data/sessions.csv'
session_features, session_data = read_csv(session_data_filename)

# organize all session data by user and generate unique sets of
#    actions, action_type, action_details, and device_types

print("Organizing og session data by user...")

sessions_by_user = dict()
distinct_actions = set()
distinct_action_types = set()
distinct_action_detail = set()
distinct_device_types = set()

for entry in session_data:
    user = entry[0]
    if user in sessions_by_user:
        sessions_by_user[user].append(entry)
    else:
        sessions_by_user[user] = [entry]

    # str replace ' ' with '_' for things like 'Windows Desktop'
    distinct_actions.add(entry[session_features['action']].replace(' ', '_'))
    distinct_action_types.add(entry[session_features['action_type']].replace(' ', '_'))
    distinct_action_detail.add(entry[session_features['action_detail']].replace(' ', '_'))
    distinct_device_types.add(entry[session_features['device_type']].replace(' ', '_'))

# for each user, initialize a dictionary for holding all session features

session_features_by_user = dict()
for user in sessions_by_user:
    session_features_by_user[user] = dict()  # probably a more elegant way to do this
                                             # but didn't want to use a defaultdict

# add in engineered features for a user's: number of actions, distinct actions,
#     distinct action_types, distinct action_details, and distinct device_types

print("Adding DISTINCT features to user feature vectors...")

session_features_by_user = add_all_and_distinct_counts_to_user_feature_vector(
    sessions_by_user, session_features, session_features_by_user)

# add in engineered features for a user's: total, average, std, max, min,
#    skew, and kurtosis for secs_elapsed

print("Adding STATISTICAL TIME features to user feature vectors...")

session_features_by_user = add_timestamp_features_to_user_feature_vector(
    sessions_by_user, session_features, session_features_by_user)

# add in engineered features for a user's ratio of time spent on each unique:
#     action, action_type, action_detail, and device_type

print("Adding TIME RATIO features to user feature vectors...")

session_features_by_user = add_total_time_for_all_unique_sessions_things_to_user_feature_vector(
    sessions_by_user, session_features, session_features_by_user,
    distinct_actions, distinct_action_types, distinct_action_detail, distinct_device_types)

# generate csv from engineered features

outfile_name = 'sessions_distinct_and_time_features.csv'
print("Writing features to %s..." % outfile_name)
write_engineered_session_csv(session_features_by_user, outfile_name)

print("Complete.")
